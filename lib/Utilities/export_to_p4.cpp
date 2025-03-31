#include "export_to_p4.h"

#include <cassert>
#include <string>
#include <system_error>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#pragma GCC diagnostic pop

#include "interleave_with_error.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR::Utilities {

namespace {

//================================================================================================//
// Utility Functions (Module Level)
//================================================================================================//
namespace {

/// Helper function to recursively add a type and its subtypes
void addTypeAndSubTypes(mlir::Type type, llvm::SetVector<mlir::Type> &collectedTypes) {
    if (!type) {
        return;
    }
    // Resolve references.
    if (auto referenceType = mlir::dyn_cast<P4HIR::ReferenceType>(type)) {
        type = referenceType.getObjectType();
    }
    if (mlir::isa<P4HIR::ErrorType, P4HIR::EnumType, P4HIR::SerEnumType, P4HIR::AliasType>(type)) {
        bool skipType = false;
        if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(type)) {
            // Action enums are an internal enum type, do not declare it.
            // TODO: Make this type more explicit?
            skipType = enumType.getName() == "action_enum";
        }
        if (!skipType) {
            collectedTypes.insert(type);
        }
    }
    if (auto structLikeType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(type)) {
        for (auto field : structLikeType.getFields()) {
            addTypeAndSubTypes(field.type, collectedTypes);
        }
        collectedTypes.insert(type);
    }
}

/// @brief Finds all MLIR types used anywhere within a ModuleOp that need to be declared in the
/// top-level of the P4 program.
/// @param moduleOp The MLIR module to scan.
/// @return A SetVector containing the unique, matching types found.
llvm::SetVector<mlir::Type> getAllTypesToDeclare(mlir::ModuleOp moduleOp) {
    llvm::SetVector<mlir::Type> collectedTypes;

    // TODO: There are a lot of duplicate iterations here and this is why we use a SetVector.
    // Does MLIR offer a convenient way to retrieve unique types for a module?
    moduleOp.walk([&](mlir::Operation *op) {
        // Collect types from block arguments
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (mlir::BlockArgument arg : block.getArguments()) {
                    addTypeAndSubTypes(arg.getType(), collectedTypes);
                }
            }
        }

        // Collect types from operation results.
        for (mlir::Type type : op->getResultTypes()) {
            addTypeAndSubTypes(type, collectedTypes);
        }

        // Collect types from operation operands.
        for (mlir::Type type : op->getOperandTypes()) {
            addTypeAndSubTypes(type, collectedTypes);
        }
    });

    return collectedTypes;
}

/// @brief Escapes a string literal for inclusion in P4 source code.
/// @param input The string to escape.
/// @return The escaped string enclosed in double quotes.
/// @example P4 Code.
/// ```p4
/// string s = "This is a \"test\" string with \\ backslashes.";
/// ```
std::string escapeP4String(llvm::StringRef input) {
    std::string result = "\"";
    for (char c : input) {
        if (c == '"') {
            result += "\\\"";
        } else if (c == '\\') {
            result += "\\\\";
        } else {
            result += c;
        }
    }
    result += "\"";
    return result;
}

/// @brief Retrieves the 'annotations' DictionaryAttr from an operation, if it exists.
/// @param op The operation to inspect.
/// @return An optional containing the DictionaryAttr if found, otherwise nullopt.
inline std::optional<mlir::DictionaryAttr> getAnnotationsAttr(mlir::Operation *op) {
    if (op == nullptr) {
        return std::nullopt;
    }
    // Centralized check using the attribute name.
    constexpr llvm::StringRef annotationAttrName = "annotations";
    if (mlir::Attribute attr = op->getAttr(annotationAttrName)) {
        if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
            return dictAttr.empty() ? std::nullopt : std::optional(dictAttr);
        }
    }
    return std::nullopt;
}

/// @brief Finds the definition operation (PackageOp, ControlOp, ParserOp, ExternOp)
///        that a given InstantiateOp refers to via its 'callee' symbol.
/// @param instantiateOp The p4hir.instantiate operation.
/// @return Success containing a pointer to the definition operation, or failure.
llvm::FailureOr<mlir::Operation *> findInstantiatedOpDefinition(
    P4HIR::InstantiateOp instantiateOp) {
    if (!instantiateOp) {
        // Returning failure requires an error emission site if we want detail.
        return mlir::failure();
    }

    mlir::SymbolRefAttr calleeAttr = instantiateOp.getCalleeAttr();
    if (!calleeAttr) {
        return mlir::failure();
    }

    mlir::Operation *symbolTableOp = instantiateOp->getParentOfType<mlir::ModuleOp>();
    if (symbolTableOp == nullptr) {
        return mlir::failure();
    }

    mlir::SymbolTableCollection symbolTable;
    mlir::Operation *definitionOp = symbolTable.lookupSymbolIn(symbolTableOp, calleeAttr);

    if (definitionOp == nullptr) {
        return mlir::failure();
    }
    return definitionOp;
}
}  // namespace

/// @brief Convets P4HIR MLIR dialect to P4 source code.
class P4HirToP4Exporter {
 public:
    /// @brief Constructs the exporter with the given options.
    /// @param options Configuration options for the export process.
    explicit P4HirToP4Exporter(P4HirToP4ExporterOptions options) : options_(options) {}

    /// @brief Gets the exporter configuration options.
    /// @return The current exporter options.
    [[nodiscard]] P4HirToP4ExporterOptions options() const { return options_; }

    /// @brief Writes the P4 program representation of the MLIR module to the stream.
    /// @param module The MLIR module to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult writeProgram(mlir::ModuleOp module, ExtendedFormattedOStream &ss) {
        // 1. Try to find all the used types, and declare all complex types.
        llvm::SetVector<mlir::Type> allTypes = getAllTypesToDeclare(module);
        for (mlir::Type type : allTypes) {
            if (failed(declareComplexType(type, ss))) {
                return mlir::failure();
            }
        }

        // 2. Generate the programmable blocks and the package.
        for (auto &op : module.getOps()) {
            if (failed(exportTopLevelStatement(op, ss))) {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

 private:
    P4HirToP4ExporterOptions options_;
    /// Tracks assignment operations handled during variable hoisting to avoid re-exporting them.
    /// TODO: Consider just returning this in hoisting?
    llvm::DenseSet<mlir::Operation *> hoistedAssignOps;

    /// @brief Exports P4 annotations based on an MLIR DictionaryAttr.
    /// @param annotations The DictionaryAttr containing annotations.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// @name("value") @flag @list("a", "b") @kv(foo=1, bar=true)
    /// ```
    mlir::LogicalResult exportAnnotations(mlir::DictionaryAttr annotations,
                                          ExtendedFormattedOStream &ss) {
        if (!annotations || annotations.empty()) {
            // No annotations to export.
            return mlir::success();
        }

        for (auto namedAttr : annotations) {
            // Emit annotation name.
            ss << "@" << namedAttr.getName().getValue();

            auto value = namedAttr.getValue();

            if (mlir::isa<mlir::UnitAttr>(value)) {
                // It's a flag annotation, e.g., @hidden - do nothing more.
                continue;
            }
            // It's a single value annotation, e.g., @name("foo") or @size(100).
            ss << "(";
            // Handle single value.
            if (failed(exportConstantAttr(value, ss))) {
                return mlir::failure();
            }
            ss << ")";
            // Space between annotations.
            ss << " ";
        }
        return mlir::success();
    }

    //===------------------------------------------------------------------===//
    // Variable Hoisting and Declaration.
    //===------------------------------------------------------------------===//

    mlir::LogicalResult declareVariable(P4HIR::VariableOp varOp, ExtendedFormattedOStream &ss) {
        // Check for immediate initialization: varOp followed by assignOp using varOp's result.
        if (auto assignOp = mlir::dyn_cast_if_present<P4HIR::AssignOp>(varOp->getNextNode())) {
            if (assignOp.getRef() == varOp.getResult()) {
                // Export combined declaration and initialization.
                // Example: bit<32> myVar = initialValue;.
                if (failed(exportP4Declaration(varOp, ss))) {
                    return mlir::failure();
                }
                ss << " = ";
                if (failed(exportExpression(assignOp.getValue(), ss))) {
                    return mlir::failure();
                }
                ss.semicolon();
                // Mark the assignment as handled so it's skipped later.
                hoistedAssignOps.insert(assignOp);
                return mlir::success();
            }
        }

        // Export declaration only.
        // Example: bit<32> myVar;
        if (failed(exportP4Declaration(varOp, ss))) {
            return mlir::failure();
        }
        ss.semicolon();
        return mlir::success();
    }

    /// @brief Collects all variables in a block, hoists their declarations to the top
    ///        of the current P4 scope, handling immediate initializations.
    /// @param block The MLIR block to scan for variables.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// {
    ///     // Variable Declarations.
    ///     bit<32> counter = 0;
    ///     bool flag;
    ///
    ///     // Original statements follow...
    ///     flag = true;
    /// }
    /// ```
    mlir::LogicalResult hoistAndDeclareVariables(mlir::Block &block, ExtendedFormattedOStream &ss) {
        llvm::SmallVector<P4HIR::VariableOp> variables;
        for (mlir::Operation &op : block) {
            if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
                variables.push_back(varOp);
            }
        }

        if (variables.empty()) {
            return mlir::success();
        }

        for (auto varOp : variables) {
            if (failed(declareVariable(varOp, ss))) {
                return mlir::failure();
            }
        }
        ss.newline();
        return mlir::success();
    }

    /// @brief Exports the operations within a single MLIR block as P4 statements,
    ///        skipping variable declarations and handled initializations.
    /// @param block The MLIR block whose operations are to be exported.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// TODO: It is possible that exportBlock is called within nestedOps. How will this impact
    /// hoisting?
    mlir::LogicalResult exportBlock(mlir::Block &block, ExtendedFormattedOStream &ss,
                                    bool useHoisting = true) {
        if (useHoisting) {
            // Hoist and declare block-local variables first.
            if (failed(hoistAndDeclareVariables(block, ss))) {
                return mlir::failure();
            }
        }
        for (mlir::Operation &op : block.getOperations()) {
            // Skip variable declarations (handled by hoisting).
            if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
                if (!useHoisting) {
                    if (failed(declareVariable(varOp, ss))) {
                        return mlir::failure();
                    }
                }
                continue;
            }
            if (hoistedAssignOps.count(&op) != 0U) {
                continue;
            }

            // Skip implicit yields at the end of blocks.
            if (mlir::isa<P4HIR::YieldOp>(op) && &op == block.getTerminator()) {
                continue;
            }

            // Determine context and dispatch to the appropriate statement exporter.
            auto *parentOp = block.getParentOp();
            mlir::LogicalResult status = mlir::failure();

            if (mlir::isa<P4HIR::ControlApplyOp, P4HIR::FuncOp, P4HIR::IfOp, P4HIR::CaseOp,
                          P4HIR::ScopeOp>(parentOp)) {
                status = exportCommonStatementWithError(op, ss);
            } else if (mlir::isa<P4HIR::ParserStateOp>(parentOp)) {
                status = exportParserStateStatement(op, ss);
            } else if (mlir::isa<P4HIR::ControlOp>(parentOp)) {
                // Statements directly inside Control (outside apply).
                status = exportControlStatement(op, ss);
            } else if (mlir::isa<P4HIR::ParserOp>(parentOp)) {
                // Statements directly inside Parser (outside state).
                status = exportParserStatement(op, ss);
            } else {
                return op.emitError() << "Unsupported parent context (" << parentOp->getName()
                                      << ") in exportBlock";
            }

            // Check status after dispatch.
            if (failed(status)) {
                return mlir::failure();
            }
        }
        return mlir::success();
    }

    //===------------------------------------------------------------------===//
    // Top-Level Declaration & Statement Export.
    //===------------------------------------------------------------------===//

    /// @brief Exports a single top-level operation from the MLIR module.
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportTopLevelStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        // Attach all annotations.
        if (auto annots = getAnnotationsAttr(&op)) {
            if (failed(exportAnnotations(*annots, ss))) {
                return mlir::failure();
            }
        }
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op)) {
            if (failed(exportTopLevelFuncOp(funcOp, ss))) {
                return mlir::failure();
            }
            ss.newline();
            return mlir::success();
        }
        if (auto overloadSetOp = mlir::dyn_cast<P4HIR::OverloadSetOp>(op)) {
            for (auto &overloadOp : overloadSetOp.getOps()) {
                if (failed(exportTopLevelStatement(overloadOp, ss))) {
                    return mlir::failure();
                }
            }
            return mlir::success();
        }
        if (auto instantiateOp = mlir::dyn_cast<P4HIR::InstantiateOp>(op)) {
            if (failed(exportTopLevelInstance(instantiateOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto packageOp = mlir::dyn_cast<P4HIR::PackageOp>(op)) {
            // Package declaration is handled by the main instantiate op.
            return mlir::success();
        }
        if (auto externOp = mlir::dyn_cast<P4HIR::ExternOp>(op)) {
            if (failed(exportExternOp(externOp, ss))) {
                return mlir::failure();
            }
            ss.newline();
            return mlir::success();
        }
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            if (failed(exportConstantDeclaration(constOp, ss))) {
                return mlir::failure();
            }
            ss.newline();
            return mlir::success();
        }
        if (mlir::isa<P4HIR::AssignOp>(op)) {
            // Should be handled by hoisting.
            return mlir::success();
        }
        if (auto uninitializedOp = mlir::dyn_cast<P4HIR::UninitializedOp>(op)) {
            // TODO: What is the use-case here?
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(exportP4Declaration(varOp, ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
            if (options().mainPackageOnly) {
                return mlir::success();
            }
            if (failed(exportParserDeclaration(parserOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
            if (options().mainPackageOnly) {
                return mlir::success();
            }
            if (failed(exportControlDeclaration(controlOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        return op.emitError() << "Unsupported top-level operation '" << op.getName() << "'";
    }

    mlir::LogicalResult exportExternMethodDefinition(P4HIR::FuncOp &funcOp,
                                                     ExtendedFormattedOStream &ss) {
        if (!funcOp.isExternal()) {
            return funcOp.emitError() << "Only extern methods are supported";
        }
        auto parentExtern = funcOp->getParentOfType<P4HIR::ExternOp>();

        if (parentExtern) {
            // If the names of the extern and method are equal, then it is an constructor.
            // Do not emit a type.
            if (parentExtern.getName() != funcOp.getName()) {
                if (failed(exportP4Type(funcOp.getFunctionType().getReturnType(), ss))) {
                    return mlir::failure();
                }
                ss << " ";
            }
        } else {
            ss << "extern ";
            if (failed(exportP4Type(funcOp.getFunctionType().getReturnType(), ss))) {
                return mlir::failure();
            }
            ss << " ";
        }

        // Handle overloaded method names within externs correctly.
        if (auto overloadSetOp = funcOp->getParentOfType<P4HIR::OverloadSetOp>()) {
            ss << overloadSetOp.getName();
        } else {
            ss << funcOp.getName();
        }

        if (failed(exportTypeArguments(funcOp.getFunctionType().getTypeArguments(), ss))) {
            return mlir::failure();
        }

        ss << "(";
        unsigned numArgs = funcOp.getNumArguments();
        for (size_t idx = 0; idx < numArgs; ++idx) {
            if (idx > 0) {
                ss << ", ";
            }

            P4HIR::ParamDirection direction = funcOp.getArgumentDirection(idx);
            if (failed(exportParameterDirection(direction, ss))) {
                return mlir::failure();
            }

            if (failed(getOriginalParamName(funcOp, idx, ss))) {
                return mlir::failure();
            }
        }
        ss << ")";
        if (funcOp.getAction() || (!funcOp.isExternal())) {
            // Emit body for actions or non-extern function definitions.
            ss << " ";
            ss.openBrace();
            // An action or defined function has a body region.
            if (!funcOp.getBody().empty()) {
                // Assuming single block for typical function bodies.
                if (failed(exportBlock(funcOp.getBody().front(), ss))) {
                    return mlir::failure();
                }
            }
            ss.closeBrace();
        } else {
            // Emit semicolon for extern method declarations or function declarations.
            ss.semicolon();
        }

        return mlir::success();
    }

    /// @brief Exports a p4hir.extern declaration.
    /// @param externOp The ExternOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// extern packet_in {
    ///     void extract<T>(out T hdr);
    ///     // ... other methods ...
    /// }
    /// ```
    mlir::LogicalResult exportExternOp(P4HIR::ExternOp &externOp, ExtendedFormattedOStream &ss) {
        ss << "extern " << externOp.getName();
        mlir::SmallVector<mlir::Type> typeParamTypes;
        if (mlir::ArrayAttr typeParamsAttr = externOp.getTypeParametersAttr()) {
            for (auto typeAttr : typeParamsAttr) {
                if (auto ta = mlir::dyn_cast<mlir::TypeAttr>(typeAttr)) {
                    typeParamTypes.push_back(ta.getValue());
                }
            }
        }
        if (failed(exportTypeArguments(typeParamTypes, ss))) {
            return mlir::failure();
        }
        ss << " ";
        ss.openBrace();
        // Externs do not typically have variable declarations inside.
        // Directly export method signatures/overloads.
        for (auto &nestedOp : externOp.getOps()) {
            if (failed(exportExternStatementOp(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports a statement within an extern definition (e.g., method signature).
    /// @param op The operation within the extern body.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportExternStatementOp(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op)) {
            if (failed(exportExternMethodDefinition(funcOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto overloadSetOp = mlir::dyn_cast<P4HIR::OverloadSetOp>(op)) {
            for (auto &overloadOp : overloadSetOp.getOps()) {
                if (failed(exportExternStatementOp(overloadOp, ss))) {
                    return mlir::failure();
                }
            }
            return mlir::success();
        }
        return op.emitError() << "Unsupported operation '" << op.getName() << "' within extern";
    }

    /// @brief Exports a top-level function or action definition.
    /// @param funcOp The FuncOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code (Action).
    /// ```p4
    /// action my_action(in bit<32> data) {
    ///     // Variable Declarations.
    ///     bool local_flag = true;
    ///
    ///     // Action body statements...
    ///     local_flag = data > 0;
    /// }
    /// ```
    /// @example P4 Code (Extern Method Declaration).
    /// ```p4
    /// extern MyExtern {
    ///     MyExtern(); // Constructor declaration.
    ///     void method(in bit<8> p); // Method declaration.
    /// }
    /// ```
    mlir::LogicalResult exportTopLevelFuncOp(P4HIR::FuncOp &funcOp, ExtendedFormattedOStream &ss) {
        if (funcOp.getAction()) {
            ss << "action ";
        } else {
            if (funcOp.isExternal() && !funcOp->getParentOfType<P4HIR::ExternOp>()) {
                ss << "extern ";
            }
            if (failed(exportP4Type(funcOp.getFunctionType().getReturnType(), ss))) {
                return mlir::failure();
            }
            ss << " ";
        }

        // Handle overloaded method names within externs correctly.
        if (auto overloadSetOp = funcOp->getParentOfType<P4HIR::OverloadSetOp>()) {
            ss << overloadSetOp.getName();
        } else {
            ss << funcOp.getName();
        }

        if (failed(exportTypeArguments(funcOp.getFunctionType().getTypeArguments(), ss))) {
            return mlir::failure();
        }

        ss << "(";
        unsigned numArgs = funcOp.getNumArguments();
        for (size_t idx = 0; idx < numArgs; ++idx) {
            if (idx > 0) {
                ss << ", ";
            }

            P4HIR::ParamDirection direction = funcOp.getArgumentDirection(idx);
            if (failed(exportParameterDirection(direction, ss))) {
                return mlir::failure();
            }

            if (failed(getOriginalParamName(funcOp, idx, ss))) {
                return mlir::failure();
            }
        }
        ss << ")";
        if (funcOp.getAction() || (!funcOp.isExternal())) {
            // Emit body for actions or non-extern function definitions.
            ss << " ";
            ss.openBrace();
            // An action or defined function has a body region.
            if (!funcOp.getBody().empty()) {
                // Assuming single block for typical function bodies.
                if (failed(exportBlock(funcOp.getBody().front(), ss))) {
                    return mlir::failure();
                }
            }
            ss.closeBrace();
        } else {
            // Emit semicolon for extern method declarations or function declarations.
            ss.semicolon();
        }

        return mlir::success();
    }

    /// @brief Exports a p4hir.parser declaration.
    /// @param parserOp The ParserOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// parser MyParser(packet_in pkt, out headers hdr) {
    ///     // Parser-local Variable Declarations.
    ///     bit<16> temp_var;
    ///
    ///     // States follow...
    ///     state start { transition parse_ethernet; }
    ///     // ... other states ...
    /// }
    /// ```
    mlir::LogicalResult exportParserDeclaration(P4HIR::ParserOp &parserOp,
                                                ExtendedFormattedOStream &ss) {
        ss << "parser " << parserOp.getName() << "(";
        if (failed(exportParameterList(parserOp, ss))) {
            return mlir::failure();
        }
        ss << ")";
        llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> ctorParamTypes;
        if (auto ctorTypeAttr = parserOp.getCtorTypeAttr()) {
            if (auto ctorType = mlir::dyn_cast<P4HIR::CtorType>(ctorTypeAttr.getValue())) {
                ctorParamTypes = ctorType.getInputs();
            } else {
                return parserOp.emitError()
                       << "parserOp has CtorTypeAttr but not a valid p4hir.CtorType.";
            }
        }
        if (ctorParamTypes.empty()) {
            ss << " ";
        } else {
            ss << "(";
            auto result = interleaveCommaWithError(ctorParamTypes, ss, [&](auto paramTuple) {
                if (failed(exportP4Type(paramTuple.second, ss))) {
                    return mlir::failure();
                }
                ss << " " << paramTuple.first.getValue();
                return mlir::success();
            });
            if (failed(result)) {
                return mlir::failure();
            }
            ss << ") ";
        }
        ss.openBrace();
        hoistedAssignOps.clear();

        // Export the remaining statements.
        if (failed(exportBlock(parserOp.getBody().front(), ss))) {
            return mlir::failure();
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports a p4hir.control declaration.
    /// @param controlOp The ControlOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// control MyControl(inout headers hdr, metadata_t meta) {
    ///     // Control-local Variable Declarations.
    ///     bit<8> control_local = 0;
    ///
    ///     // Actions, tables, apply block follow...
    ///     action my_action() { /* ... */ }
    ///     table my_table { /* ... */ }
    ///     apply {
    ///         // Apply block statements...
    ///     }
    /// }
    /// ```
    mlir::LogicalResult exportControlDeclaration(P4HIR::ControlOp &controlOp,
                                                 ExtendedFormattedOStream &ss) {
        ss << "control " << controlOp.getName() << "(";
        if (failed(exportParameterList(controlOp, ss))) {
            return mlir::failure();
        }
        ss << ")";
        llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> ctorParamTypes;
        if (auto ctorTypeAttr = controlOp.getCtorTypeAttr()) {
            if (auto ctorType = mlir::dyn_cast<P4HIR::CtorType>(ctorTypeAttr.getValue())) {
                ctorParamTypes = ctorType.getInputs();
            } else {
                return controlOp.emitError()
                       << "controlOp has CtorTypeAttr but not a valid p4hir.CtorType.";
            }
        }
        if (ctorParamTypes.empty()) {
            ss << " ";
        } else {
            ss << "(";
            auto result = interleaveCommaWithError(ctorParamTypes, ss, [&](auto paramTuple) {
                if (failed(exportP4Type(paramTuple.second, ss))) {
                    return mlir::failure();
                }
                ss << " " << paramTuple.first.getValue();
                return mlir::success();
            });
            if (failed(result)) {
                return mlir::failure();
            }
            ss << ") ";
        }
        ss.openBrace();
        hoistedAssignOps.clear();

        // Export the remaining statements.
        if (failed(exportBlock(controlOp.getBody().front(), ss))) {
            return mlir::failure();
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports a p4hir.const declaration.
    /// @param constant The ConstOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// const bit<8> MY_CONST = 8w10;
    /// const bool ENABLED = true;
    /// ```
    mlir::LogicalResult exportConstantDeclaration(P4HIR::ConstOp &constant,
                                                  ExtendedFormattedOStream &ss) {
        ss << "const ";
        if (failed(exportP4Type(constant.getType(), ss))) {
            return mlir::failure();
        }
        ss << " ";
        if (auto name = constant.getName()) {
            ss << name.value();
        } else {
            // Handle anonymous constants if necessary, maybe generate a name.
            return constant.emitError() << "Named constant declaration requires a name.";
        }
        ss << " = ";
        auto valueAttr = constant.getValueAttr();
        if (failed(exportConstantAttr(valueAttr, ss))) {
            return mlir::failure();
        }
        ss.semicolon();
        return mlir::success();
    }

    /// @brief Exports a p4hir.package declaration (the signature part).
    /// @param packageOp The PackageOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// package MyPackage<ParserType, ControlType>(ParserType p, ControlType c);
    /// ```
    mlir::LogicalResult exportPackageOp(P4HIR::PackageOp &packageOp, ExtendedFormattedOStream &ss) {
        llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> ctorParamTypes;
        if (auto ctorTypeAttr = packageOp.getCtorTypeAttr()) {
            if (auto ctorType = mlir::dyn_cast<P4HIR::CtorType>(ctorTypeAttr.getValue())) {
                ctorParamTypes = ctorType.getInputs();
            } else {
                return packageOp.emitError()
                       << "packageOp has CtorTypeAttr but not a valid p4hir.CtorType.";
            }
        }

        // Export programmable block declarations needed by the package signature.
        for (const auto &ctorParamType : ctorParamTypes) {
            mlir::Type paramType = ctorParamType.second;
            if (llvm::failed(exportProgrammableBlock(paramType, ss))) {
                return mlir::failure();
            }
        }

        ss << "package " << packageOp.getName();
        // Export type parameters <...>.
        mlir::SmallVector<mlir::Type> typeParamTypes;
        if (mlir::ArrayAttr typeParamsAttr = packageOp.getTypeParametersAttr()) {
            for (auto typeAttr : typeParamsAttr) {
                if (auto ta = mlir::dyn_cast<mlir::TypeAttr>(typeAttr)) {
                    typeParamTypes.push_back(ta.getValue());
                }
            }
        }
        if (failed(exportTypeArguments(typeParamTypes, ss))) {
            return mlir::failure();
        }
        ss << "(";

        // Export constructor parameters (...).
        mlir::ArrayAttr ctorArgAttrs = packageOp.getArgAttrsAttr();
        for (size_t idx = 0; idx < ctorParamTypes.size(); ++idx) {
            if (idx > 0) {
                ss << ", ";
            }
            mlir::Type paramType = ctorParamTypes[idx].second;

            // Get the P4 string for the parameter type instance (e.g., "MyParser<H, M>").
            if (failed(exportP4Type(paramType, ss))) {
                return mlir::failure();
            }
            // Get the parameter name.
            if (ctorArgAttrs && idx < ctorArgAttrs.size()) {
                if (auto dictAttr =
                        mlir::dyn_cast_if_present<mlir::DictionaryAttr>(ctorArgAttrs[idx])) {
                    if (auto nameAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(
                            dictAttr.get(P4HIR::FuncOp::getParamNameAttrName()))) {
                        if (!nameAttr.getValue().empty()) {
                            ss << " " << nameAttr.getValue();
                        } else {
                            // Fallback or error if name is missing but expected.
                            return packageOp.emitError()
                                   << "Missing name for constructor parameter " << idx;
                        }
                    } else {
                        return packageOp.emitError()
                               << "Missing/invalid name attribute for constructor parameter "
                               << idx;
                    }
                } else {
                    return packageOp.emitError()
                           << "Constructor argument attribute " << idx << " is not a dictionary.";
                }
            } else {
                // Use the name from the CtorType definition if arg_attrs are missing.
                ss << " " << ctorParamTypes[idx].first.getValue();
            }
        }
        ss << ")";
        ss.semicolon();
        return mlir::success();
    }

    /// @brief Exports a p4hir.instantiate operation, focusing on the main package instantiation.
    /// @param instantiateOp The InstantiateOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// // Assuming MyParser, MyControl declared/defined earlier.
    /// MyParser() p;
    /// MyControl() c;
    /// MyPackage(p, c) main; // Instantiation.
    /// ```
    mlir::LogicalResult exportTopLevelInstance(P4HIR::InstantiateOp instantiateOp,
                                               ExtendedFormattedOStream &ss) {
        auto definitionOpt = findInstantiatedOpDefinition(instantiateOp);
        if (failed(definitionOpt)) {
            // Attempt to emit the error at the op's location.
            return instantiateOp.emitError() << "Could not resolve definition for symbol '"
                                             << instantiateOp.getCallee() << "'.";
        }
        auto *definition = *definitionOpt;

        // Only emit the main package instantiation itself.
        if (auto packageDef = mlir::dyn_cast<P4HIR::PackageOp>(definition)) {
            auto resultType = instantiateOp.getResult().getType();
            auto packageType = mlir::dyn_cast<P4HIR::PackageType>(resultType);
            if (!packageType) {
                return instantiateOp.emitError()
                       << "InstantiateOp result is not a PackageType, but definition is PackageOp.";
            }

            // First, ensure the package *signature* is declared.
            if (failed(exportPackageOp(packageDef, ss))) {
                return mlir::failure();
            }

            // Now, emit the instantiation line.
            ss << packageType.getName() << "(";

            // Export the constructor arguments (which are likely instances themselves).
            if (failed(Utilities::interleaveCommaWithError(
                    instantiateOp.getArgOperands(), ss, [&](mlir::Value arg) {
                        // Arguments to package constructor are typically instances.
                        if (auto argInstOp = arg.getDefiningOp<P4HIR::InstantiateOp>()) {
                            // Export the *name* of the instance being passed.
                            ss << argInstOp.getName();
                            // TODO: What about arguments here?.
                            ss << "()";
                            return mlir::success();
                        }
                        // Handle constants or other direct values if allowed by P4 spec.
                        return exportExpression(arg, ss);
                    }))) {
                return mlir::failure();
            }
            ss << ") ";
            ss << instantiateOp.getName();
            ss.semicolon();
            ss.newline();
            return mlir::success();
        }
        if (mlir::isa<P4HIR::ParserOp, P4HIR::ControlOp, P4HIR::ExternOp>(definition)) {
            if (!options().mainPackageOnly) {
                // TODO: How to handle this?.
            }
            // If mainPackageOnly, assume the package instantiation handles wiring these up.
            return mlir::success();
        }

        return instantiateOp.emitError() << "Unexpected definition type (" << definition->getName()
                                         << ") resolved for instantiation.";
    }

    /// @brief Exports the declaration signature of a programmable block (Parser or Control).
    /// @param type The ParserType or ControlType to declare.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// parser MyParser<H, M>(packet_in p, out H parsed_hdrs); // Declaration only.
    /// control MyControl<T>(inout T data); // Declaration only.
    /// ```
    mlir::LogicalResult exportProgrammableBlock(mlir::Type &type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::ParserType>([&](P4HIR::ParserType parserType) {
                ss << P4HIR::ParserType::getMnemonic() << " ";
                ss << parserType.getName();
                if (failed(exportTypeArguments(parserType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }

                ss << "(";
                if (failed(Utilities::interleaveCommaWithError(
                        parserType.getInputs(), ss, [&](mlir::Type inputType) {
                            // Need parameter names here? P4 allows declarations without names.
                            // For now, just export type. Full exportParameterList needs names.
                            return exportP4Type(inputType, ss);
                        }))) {
                    return mlir::failure();
                }
                ss << ")";
                ss.semicolon();
                return mlir::success();
            })
            .Case<P4HIR::ControlType>([&](P4HIR::ControlType controlType) {
                ss << P4HIR::ControlType::getMnemonic() << " ";
                ss << controlType.getName();
                if (failed(exportTypeArguments(controlType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }

                ss << "(";
                if (failed(Utilities::interleaveCommaWithError(
                        controlType.getInputs(), ss, [&](mlir::Type inputType) {
                            // See comment in ParserType case.
                            return exportP4Type(inputType, ss);
                        }))) {
                    return mlir::failure();
                }
                ss << ")";
                ss.semicolon();
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                auto loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc)
                       << "Unsupported P4HIR type for programmable block declaration: " << t;
            });
    }

    //===========================================================================================//
    // Type Exporting.
    //===========================================================================================//

    /// @brief Exports the P4 type name corresponding to an MLIR type.
    /// @param type The MLIR type to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code (various outputs).
    /// ```p4
    /// bit<32> // from !p4hir.bit<32>.
    /// bool    // from !p4hir.bool.
    /// MyHeader // from !p4hir.header<"MyHeader", ...>.
    /// MyExtern<W> // from !p4hir.extern<"MyExtern", !p4hir.bit<W>>.
    /// ```
    mlir::LogicalResult exportP4Type(mlir::Type type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            // Handle reference types by exporting the underlying object type.
            .Case<P4HIR::ReferenceType>([&](P4HIR::ReferenceType ref) {
                // Recurse and resolve the reference.
                return exportP4Type(ref.getObjectType(), ss);
            })
            // Struct-like types just use their name.
            .Case<P4HIR::HeaderType>([&](P4HIR::HeaderType headerType) {
                ss << headerType.getName();
                return mlir::success();
            })
            .Case<P4HIR::StructType>([&](P4HIR::StructType structType) {
                ss << structType.getName();
                return mlir::success();
            })
            .Case<P4HIR::HeaderUnionType>([&](P4HIR::HeaderUnionType unionType) {
                ss << unionType.getName();
                return mlir::success();
            })
            // Base types.
            .Case<P4HIR::BitsType>([&](P4HIR::BitsType bitType) {
                ss << (bitType.isSigned() ? "int" : "bit") << "<" << bitType.getWidth() << ">";
                return mlir::success();
            })
            .Case<P4HIR::BoolType>([&](P4HIR::BoolType) {
                ss << P4HIR::BoolType::getMnemonic();
                return mlir::success();
            })
            .Case<P4HIR::StringType>([&](P4HIR::StringType) {
                ss << P4HIR::StringType::getMnemonic();
                return mlir::success();
            })
            .Case<P4HIR::VoidType>([&](P4HIR::VoidType) {
                ss << P4HIR::VoidType::getMnemonic();
                return mlir::success();
            })
            // Enums and Errors use their names (or fixed 'error').
            .Case<P4HIR::EnumType>([&](P4HIR::EnumType enumType) {
                ss << enumType.getName();
                return mlir::success();
            })
            .Case<P4HIR::SerEnumType>([&](P4HIR::SerEnumType serEnumType) {
                ss << serEnumType.getName();
                return mlir::success();
            })
            .Case<P4HIR::ErrorType>([&](P4HIR::ErrorType) {
                // The type name for error is just 'error' in P4.
                ss << P4HIR::ErrorType::getMnemonic();
                return mlir::success();
            })
            // Type variable name.
            .Case<P4HIR::TypeVarType>([&](P4HIR::TypeVarType typeVarType) {
                ss << typeVarType.getName();
                return mlir::success();
            })
            // Control/Parser/Package/Extern types (potentially with type args).
            .Case<P4HIR::ExternType>([&](P4HIR::ExternType externType) {
                ss << externType.getName();
                if (failed(exportTypeArguments(externType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }
                return mlir::success();
            })
            .Case<P4HIR::ParserType>([&](P4HIR::ParserType parserType) {
                ss << parserType.getName();
                if (failed(exportTypeArguments(parserType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }
                return mlir::success();
            })
            .Case<P4HIR::ControlType>([&](P4HIR::ControlType controlType) {
                ss << controlType.getName();
                if (failed(exportTypeArguments(controlType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }
                return mlir::success();
            })
            .Case<P4HIR::PackageType>([&](P4HIR::PackageType packageType) {
                ss << packageType.getName();
                if (failed(exportTypeArguments(packageType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }
                return mlir::success();
            })
            .Case<P4HIR::AliasType>([&](P4HIR::AliasType aliasType) {
                // Export the alias name.
                ss << aliasType.getName();
                return mlir::success();
            })
            // Array types (usually header stacks).
            .Case<P4HIR::ArrayType>([&](P4HIR::ArrayType arrayType) {
                // P4 standard arrays are still header stacks.
                // Syntactically, a fixed-size array declaration looks like: H h[N];
                // Exporting the type name refers to the *element* type.
                if (failed(exportP4Type(arrayType.getElementType(), ss))) {
                    return mlir::failure();
                }
                // The size [N] is part of the variable declaration, not the type name itself.
                return mlir::success();
            })
            .Case<P4HIR::InfIntType>([&](P4HIR::InfIntType) {
                // P4 arbitrary precision integer type.
                ss << "int";
                return mlir::success();
            })
            .Case<P4HIR::VarBitsType>([&](P4HIR::VarBitsType varBitsType) {
                ss << "varbit<" << varBitsType.getMaxWidth() << ">";
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                auto loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Unsupported P4HIR type for P4 export: " << t;
            });
    }

    /// @brief Exports the field declarations within a P4 struct, header, or union definition.
    /// @param fields An array of FieldInfo describing the fields.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code (inside a struct/header body).
    /// ```p4
    ///   bit<48> dstAddr;
    ///   bit<16> etherType;
    /// ```
    mlir::LogicalResult declareStructTypeFields(llvm::ArrayRef<P4HIR::FieldInfo> fields,
                                                ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        for (auto field : fields) {
            // The valid bit is implicit in P4 headers, skip its declaration.
            if (mlir::isa<P4HIR::ValidBitType>(field.type)) {
                continue;
            }
            if (failed(exportP4Type(field.type, ss))) {
                return mlir::failure();
            }
            ss << " " << field.name.strref();
            ss.semicolon();
        }
        return mlir::success();
    }

    mlir::LogicalResult declareErrorType(P4HIR::ErrorType errorType, ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        ss << P4HIR::ErrorType::getMnemonic() << " ";
        ss.openBrace();
        auto numFields = errorType.getFields().size();
        auto fields = errorType.getFields();
        size_t idx = 0;
        for (auto field : fields) {
            if (failed(exportConstantAttr(field, ss, false))) {
                return mlir::failure();
            }
            if (idx < numFields - 1) {
                ss << ",";
            }
            ss.newline();
            idx++;
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult declareEnumType(P4HIR::EnumType enumType, ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        ss << P4HIR::EnumType::getMnemonic() << " " << enumType.getName() << " ";
        ss.openBrace();
        auto numFields = enumType.getFields().size();
        auto fields = enumType.getFields();
        size_t idx = 0;
        for (auto field : fields) {
            if (failed(exportConstantAttr(field, ss, false))) {
                return mlir::failure();
            }
            /// TODO: Would be nice if P4 supported trailing commas...
            if (idx < numFields - 1) {
                ss << ",";
            }
            ss.newline();
            idx++;
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult declareSerEnumType(P4HIR::SerEnumType serEnumType,
                                           ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        ss << P4HIR::EnumType::getMnemonic() << " ";
        if (failed(exportP4Type(serEnumType.getType(), ss))) {
            return mlir::failure();
        }
        ss << " " << serEnumType.getName() << " ";
        ss.openBrace();
        auto numFields = serEnumType.getFields().size();
        auto fields = serEnumType.getFields();
        size_t idx = 0;
        for (auto field : fields) {
            ss << field.getName().strref() << " = ";
            auto attr = field.getValue();
            if (failed(exportConstantAttr(attr, ss, false))) {
                return mlir::failure();
            }
            /// TODO: Would be nice if P4 supported trailing commas...
            if (idx < numFields - 1) {
                ss << ",";
            }
            ss.newline();
            idx++;
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult declareStructLikeType(P4HIR::StructLikeTypeInterface structLikeType,
                                              ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        if (auto headerType = mlir::dyn_cast<P4HIR::HeaderType>(structLikeType)) {
            ss << P4HIR::HeaderType::getMnemonic() << " " << headerType.getName() << " ";
        } else if (auto structType = mlir::dyn_cast<P4HIR::StructType>(structLikeType)) {
            ss << P4HIR::StructType::getMnemonic() << " " << structType.getName() << " ";
        } else if (auto unionType = mlir::dyn_cast<P4HIR::HeaderUnionType>(structLikeType)) {
            ss << P4HIR::HeaderUnionType::getMnemonic() << " " << unionType.getName() << " ";
        } else {
            return mlir::emitError(mlir::UnknownLoc::get(structLikeType.getContext()))
                   << "Unsupported P4HIR type for P4 export: " << structLikeType;
        }
        ss.openBrace();
        if (failed(declareStructTypeFields(structLikeType.getFields(), ss))) {
            return mlir::failure();
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports the P4 definition of a struct-like type (struct, header, header_union).
    /// @param type The struct-like MLIR type to declare.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code (Header).
    /// ```p4
    /// header Ethernet_h {
    ///     bit<48> dstAddr;
    ///     bit<48> srcAddr;
    ///     bit<16> etherType;
    /// }
    /// ```
    mlir::LogicalResult declareComplexType(mlir::Type &type, ExtendedFormattedOStream &ss) {
        // TODO: Annotations.
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::StructLikeTypeInterface>(
                [&](auto structLikeType) { return declareStructLikeType(structLikeType, ss); })
            .Case<P4HIR::ErrorType>([&](auto errorType) { return declareErrorType(errorType, ss); })
            .Case<P4HIR::EnumType>([&](auto enumType) { return declareEnumType(enumType, ss); })
            .Case<P4HIR::SerEnumType>(
                [&](auto serEnumType) { return declareSerEnumType(serEnumType, ss); })
            .Case<P4HIR::AliasType>([&](auto aliasType) {
                ss << "type ";
                if (failed(exportP4Type(aliasType.getAliasedType(), ss))) {
                    return mlir::failure();
                }
                ss << " " << aliasType.getName();
                ss.semicolon();
                return mlir::success();
            })
            .Default([&](auto t) {
                auto loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Unsupported P4HIR type for declaration: " << t;
            });
    };

    /// @brief Exports P4 type arguments in angle brackets.
    /// @param types An array of MLIR types to export as arguments.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// <bit<32>, MyHeader>
    /// <> // (or nothing if empty, handled by this function).
    /// ```
    mlir::LogicalResult exportTypeArguments(mlir::ArrayRef<mlir::Type> types,
                                            ExtendedFormattedOStream &ss) {
        if (types.empty()) {
            return mlir::success();
        }
        ss << "<";
        auto result = Utilities::interleaveCommaWithError(
            types, ss, [&](mlir::Type type) { return exportP4Type(type, ss); });
        if (failed(result)) {
            return mlir::failure();
        }
        ss << ">";
        return mlir::success();
    }

    /// @brief Exports P4 type arguments represented by MLIR attributes (potentially types).
    /// @param attrs An array of MLIR attributes to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code (assuming attributes represent types).
    /// ```p4
    /// <MyStruct, int<16>>
    /// ```
    mlir::LogicalResult exportTypeAttributes(mlir::ArrayRef<mlir::Attribute> attrs,
                                             ExtendedFormattedOStream &ss) {
        if (attrs.empty()) {
            return mlir::success();
        }
        ss << "<";
        auto result = Utilities::interleaveCommaWithError(attrs, ss, [&](mlir::Attribute attr) {
            if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
                return exportP4Type(typeAttr.getValue(), ss);
            }
            return mlir::LogicalResult(mlir::emitError(mlir::UnknownLoc::get(attr.getContext()))
                                       << "Unsupported attribute kind in type arguments: " << attr
                                       << ".");
        });
        if (failed(result)) {
            return mlir::failure();
        }
        ss << ">";
        return mlir::success();
    }

    //===========================================================================================//
    // Expression Exporting.
    //===========================================================================================//

    mlir::LogicalResult exportExpression(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        // Delegate to specific operation handlers.
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            auto valueAttr = constOp.getValueAttr();
            return exportConstantAttr(valueAttr, ss);
        }
        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
            // Reading the value *from* a variable reference. Export the L-Value being read.
            return exportLValue(readOp.getRef(), ss);
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            // Handles using the *reference* itself (e.g., passing to out/inout param).
            if (auto varName = varOp.getName()) {
                ss << varName.value();
                return mlir::success();
            }
            return varOp.emitError() << "VariableOp used in expression context has no name.";
        }
        if (auto uninitializedOp = mlir::dyn_cast<P4HIR::UninitializedOp>(op)) {
            ss << "_";
            return mlir::success();
        }
        if (auto castOp = mlir::dyn_cast<P4HIR::CastOp>(op)) {
            // P4 cast syntax: (target_type) source_expression.
            ss << "(";
            if (failed(exportP4Type(castOp.getResult().getType(), ss))) {
                return mlir::failure();
            }
            ss << ")(";
            // Parenthesize source.
            if (failed(exportExpression(castOp.getSrc(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto unaryOp = mlir::dyn_cast<P4HIR::UnaryOp>(op)) {
            auto opStr = getUnaryOpString(unaryOp.getKind());
            if (failed(opStr)) {
                return unaryOp.emitError() << "Invalid unary operator kind.";
            }
            // Parenthesize for precedence.
            ss << "(" << *opStr;
            if (failed(exportExpression(unaryOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto binOp = mlir::dyn_cast<P4HIR::BinOp>(op)) {
            auto opStr = getBinaryOpString(binOp.getKind());
            if (failed(opStr)) {
                return binOp.emitError() << "Invalid binary operator kind.";
            }
            // Parenthesize for precedence.
            ss << "(";
            if (failed(exportExpression(binOp.getLhs(), ss))) {
                return mlir::failure();
            }
            ss << " " << *opStr << " ";
            if (failed(exportExpression(binOp.getRhs(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto concatOp = mlir::dyn_cast<P4HIR::ConcatOp>(op)) {
            // P4 concatenation syntax: lhs ++ rhs.
            // TODO: WHy is this not a binary operator?
            // Parenthesize for precedence.
            ss << "(";
            if (failed(exportExpression(concatOp.getLhs(), ss))) {
                return mlir::failure();
            }
            ss << " ++ ";
            if (failed(exportExpression(concatOp.getRhs(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto shlOp = mlir::dyn_cast<P4HIR::ShlOp>(op)) {
            // Parenthesize for precedence.
            ss << "(";
            if (failed(exportExpression(shlOp.getLhs(), ss))) {
                return mlir::failure();
            }
            ss << " << ";
            if (failed(exportExpression(shlOp.getRhs(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto shrOp = mlir::dyn_cast<P4HIR::ShrOp>(op)) {
            // Parenthesize for precedence.
            ss << "(";
            if (failed(exportExpression(shrOp.getLhs(), ss))) {
                return mlir::failure();
            }
            ss << " >> ";
            if (failed(exportExpression(shrOp.getRhs(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto cmpOp = mlir::dyn_cast<P4HIR::CmpOp>(op)) {
            // Need to special-case here to handle validity attributes.
            // TODO: Better way to handle this?
            if (auto *rhs = cmpOp.getRhs().getDefiningOp()) {
                if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(rhs)) {
                    if (auto validityAttr =
                            mlir::dyn_cast<P4HIR::ValidityBitAttr>(constOp.getValueAttr())) {
                        if (validityAttr.getValue() == P4HIR::ValidityBit::Invalid) {
                            ss << "!";
                        }
                        if (failed(exportExpression(cmpOp.getLhs(), ss))) {
                            return mlir::failure();
                        }
                        ss << ".isValid()";
                        return mlir::success();
                    }
                }
            }
            auto opStr = getCompareOpString(cmpOp.getKind());
            if (failed(opStr)) {
                return cmpOp.emitError() << "Invalid compare operator kind.";
            }
            // Parenthesize for precedence.
            ss << "(";
            if (failed(exportExpression(cmpOp.getLhs(), ss))) {
                return mlir::failure();
            }
            ss << " " << *opStr << " ";
            if (failed(exportExpression(cmpOp.getRhs(), ss))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto structExtractOp = mlir::dyn_cast<P4HIR::StructExtractOp>(op)) {
            // P4 syntax: struct_instance.field_name.
            if (failed(exportExpression(structExtractOp.getInput(), ss))) {
                return mlir::failure();
            }
            if (structExtractOp.getFieldName() == P4HIR::HeaderType::validityBit) {
                // TODO: Do we need to handle validity attributes with `isValid()`?
            } else {
                ss << "." << structExtractOp.getFieldName();
            }
            return mlir::success();
        }
        if (auto sliceOp = mlir::dyn_cast<P4HIR::SliceOp>(op)) {
            // P4 syntax: expression[high:low].
            if (failed(exportExpression(sliceOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << "[" << sliceOp.getHighBit() << ":" << sliceOp.getLowBit() << "]";
            return mlir::success();
        }
        if (auto sliceRefOp = mlir::dyn_cast<P4HIR::SliceRefOp>(op)) {
            // P4 syntax: expression[high:low].
            if (failed(exportExpression(sliceRefOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << "[" << sliceRefOp.getHighBit() << ":" << sliceRefOp.getLowBit() << "]";
            return mlir::success();
        }
        if (auto callOp = mlir::dyn_cast<P4HIR::CallOp>(op)) {
            // Direct function call expression.
            ss << callOp.getCalleeAttr().getLeafReference().getValue() << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    callOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto callMethodOp = mlir::dyn_cast<P4HIR::CallMethodOp>(op)) {
            // Extern method call expression: extern_instance.method<type_args>(args).
            // Export extern instance.
            if (failed(exportExpression(callMethodOp.getBase(), ss))) {
                return mlir::failure();
            }
            // Method name.
            ss << "." << callMethodOp.getCallee().getLeafReference().getValue();
            // Type arguments <T, U>.
            if (auto typeOperands = callMethodOp.getTypeOperandsAttr()) {
                if (failed(exportTypeAttributes(typeOperands.getValue(), ss))) {
                    return mlir::failure();
                }
            }
            ss << "(";
            // Export arguments.
            if (failed(Utilities::interleaveCommaWithError(
                    callMethodOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }
        if (auto tableApplyOp = mlir::dyn_cast<P4HIR::TableApplyOp>(op)) {
            // Table apply expression: table_instance.apply().
            ss << tableApplyOp.getCallee().getLeafReference().strref() << ".apply()";
            return mlir::success();
        }
        if (auto ternaryOp = mlir::dyn_cast<P4HIR::TernaryOp>(op)) {
            // P4 syntax: (condition ? true_expr : false_expr).
            return exportTernaryOp(ternaryOp, ss);
        }
        if (auto arrayGetOp = mlir::dyn_cast<P4HIR::ArrayGetOp>(op)) {
            // Array/header stack element access: array_instance[index].
            if (failed(exportExpression(arrayGetOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << "[";
            if (failed(exportExpression(arrayGetOp.getIndex(), ss))) {
                return mlir::failure();
            }
            ss << "]";
            return mlir::success();
        }
        if (auto structOp = mlir::dyn_cast<P4HIR::StructOp>(op)) {
            // Struct constructor expression: (StructType) { field1 = val1, ... }.
            ss << "(";
            if (failed(exportP4Type(structOp.getType(), ss))) {
                return mlir::failure();
            }
            ss << ") {";
            auto iface =
                mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(structOp.getResult().getType());
            if (!iface) {
                return structOp.emitError() << "Type is not a struct-like type.";
            }
            auto fields = iface.getFields();
            auto inputs = structOp.getInput();
            if (fields.size() != inputs.size()) {
                return structOp.emitError()
                       << "Mismatch between fields in type and inputs in StructOp.";
            }
            for (size_t idx = 0; idx < fields.size(); ++idx) {
                if (mlir::isa<P4HIR::ValidBitType>(fields[idx].type)) {
                    continue;
                }
                if (idx > 0) {
                    ss << ", ";
                }
                ss << fields[idx].name.strref() << " = ";
                if (failed(exportExpression(inputs[idx], ss))) {
                    return mlir::failure();
                }
            }
            ss << "}";
            return mlir::success();
        }
        if (auto instantiateOp = mlir::dyn_cast<P4HIR::InstantiateOp>(op)) {
            // Export the instance name (e.g., " my_counter_instance").
            // TODO: Do we also need to include type instantiation or just the reference?
            ss << instantiateOp.getName();
            // Export the constructor arguments (e.g., "(1024)").
            ss << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    instantiateOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            return mlir::success();
        }

        return op.emitError() << "Unsupported operation '" << op.getName()
                              << "' in exportExpression";
    }

    /// @brief Exports an MLIR Value used as a P4 expression.
    /// @param value The MLIR Value to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportExpression(mlir::Value value, ExtendedFormattedOStream &ss) {
        mlir::Operation *op = value.getDefiningOp();
        auto loc = (op != nullptr) ? op->getLoc() : mlir::UnknownLoc::get(value.getContext());

        // Handle Block Arguments (Parameters/Loop Variables).
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            return exportParameterReference(blockArg, ss);
        }

        if (op == nullptr) {
            return mlir::emitError(loc) << "Value used in expression has no defining operation and "
                                           "is not a block argument: "
                                        << value;
        }
        return exportExpression(*op, ss);
    }

    mlir::LogicalResult exportTernaryOp(P4HIR::TernaryOp &ternaryOp, ExtendedFormattedOStream &ss) {
        // Parenthesize for precedence.
        ss << "(";
        // 1. Export the condition.
        if (failed(exportExpression(ternaryOp.getCond(), ss))) {
            return mlir::failure();
        }

        ss << " ? ";

        // 2. Export the true expression (value yielded from trueRegion).
        auto &trueRegion = ternaryOp.getTrueRegion();
        if (trueRegion.empty() || !trueRegion.hasOneBlock()) {
            return ternaryOp.emitError()
                   << "TernaryOp trueRegion must have exactly one block for P4 export.";
        }
        mlir::Block &trueBlock = trueRegion.front();
        auto trueYield = mlir::dyn_cast_if_present<P4HIR::YieldOp>(trueBlock.getTerminator());
        if (!trueYield) {
            return ternaryOp.emitError()
                   << "TernaryOp trueRegion block must terminate with p4hir.yield.";
        }
        if (trueYield.getNumOperands() != 1) {
            return trueYield.emitError()
                   << "TernaryOp trueRegion yield must have exactly one operand.";
        }
        // Parenthesize the sub-expression.
        ss << "(";
        if (failed(exportExpression(trueYield.getOperand(0), ss))) {
            return mlir::failure();
        }
        ss << ")";

        ss << " : ";

        // 3. Export the false expression (value yielded from falseRegion).
        auto &falseRegion = ternaryOp.getFalseRegion();
        if (falseRegion.empty() || !falseRegion.hasOneBlock()) {
            return ternaryOp.emitError()
                   << "TernaryOp falseRegion must have exactly one block for P4 export.";
        }
        mlir::Block &falseBlock = falseRegion.front();
        auto falseYield = mlir::dyn_cast_if_present<P4HIR::YieldOp>(falseBlock.getTerminator());
        if (!falseYield) {
            return ternaryOp.emitError()
                   << "TernaryOp falseRegion block must terminate with p4hir.yield.";
        }
        if (falseYield.getNumOperands() != 1) {
            return falseYield.emitError()
                   << "TernaryOp falseRegion yield must have exactly one operand.";
        }
        // Parenthesize the sub-expression.
        ss << "(";
        if (failed(exportExpression(falseYield.getOperand(0), ss))) {
            return mlir::failure();
        }
        ss << ")";

        // Close the outer parenthesis for the whole ternary expression.
        ss << ")";
        return mlir::success();
    }

    /// @brief Exports a P4 constant value from an MLIR attribute.
    /// @param constantAttr The MLIR attribute representing the constant value.
    /// @param ss The output stream.
    /// @param escapeStrings Escape string attributes.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// 8w10    // from #p4hir.int<10> : !p4hir.bit<8>.
    /// true    // from #p4hir.bool<true>.
    /// "hello" // from "hello".
    /// ```
    mlir::LogicalResult exportConstantAttr(mlir::Attribute &constantAttr,
                                           ExtendedFormattedOStream &ss,
                                           bool escapeStrings = true) {
        auto loc = mlir::UnknownLoc::get(constantAttr.getContext());
        if (auto intAttr = mlir::dyn_cast<P4HIR::IntAttr>(constantAttr)) {
            auto type = intAttr.getType();
            // Resolve any aliased type first. We also need to make sure to cast.
            if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type)) {
                ss << "(" << aliasType.getName() << ") ";
                type = aliasType.getAliasedType();
            }
            if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(type)) {
                unsigned width = bitsType.getWidth();
                const llvm::APInt &value = intAttr.getValue();
                ss << width << (bitsType.isSigned() ? "s" : "w");
                std::string decimalString;
                llvm::raw_string_ostream decimalStream(decimalString);
                value.print(decimalStream, /*isSigned=*/false);
                ss << decimalStream.str();
                return mlir::success();
            }
            if (auto intType = mlir::dyn_cast<P4HIR::InfIntType>(type)) {
                ss << intAttr.getValue();
                return mlir::success();
            }
            return mlir::emitError(loc) << "Unsupported type in IntAttr: " << intAttr.getType();
        }
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constantAttr)) {
            if (escapeStrings) {
                ss << escapeP4String(strAttr.getValue());
            } else {
                ss << strAttr.getValue();
            }
            return mlir::success();
        }
        if (auto boolAttr = mlir::dyn_cast<P4HIR::BoolAttr>(constantAttr)) {
            ss << (boolAttr.getValue() ? "true" : "false");
            return mlir::success();
        }
        if (auto enumAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(constantAttr)) {
            // P4 uses dot notation: EnumTypeName.FieldName.
            if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(enumAttr.getType())) {
                ss << enumType.getName() << "." << enumAttr.getField().strref();
                return mlir::success();
            }
            if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(enumAttr.getType())) {
                ss << serEnumType.getName() << "." << enumAttr.getField().strref();
                return mlir::success();
            }
            return mlir::emitError(loc)
                   << "EnumFieldAttr requires EnumType, found " << enumAttr.getType();
        }
        if (auto errorAttr = mlir::dyn_cast<P4HIR::ErrorCodeAttr>(constantAttr)) {
            // P4 error constants: error.ErrorName.
            ss << "error." << errorAttr.getField().strref();
            return mlir::success();
        }
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantAttr)) {
            // Export based on the underlying MLIR integer type (e.g., i64).
            ss << intAttr.getValue();
            return mlir::success();
        }
        if (auto symRefAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(constantAttr)) {
            // Export the leaf name for symbol references within annotations.
            ss << symRefAttr.getLeafReference().getValue();
            return mlir::success();
        }
        if (auto matchKindAttr = mlir::dyn_cast<P4HIR::MatchKindAttr>(constantAttr)) {
            // Match kinds are typically identifiers in P4.
            ss << matchKindAttr.getValue().getValue();
            return mlir::success();
        }
        if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(constantAttr)) {
            return Utilities::interleaveCommaWithError(
                arrayAttr.getValue(), ss, [&](auto elem) { return exportConstantAttr(elem, ss); });
        }
        if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(constantAttr)) {
            return Utilities::interleaveCommaWithError(
                dictAttr.getValue(), ss, [&](auto namedAttr) {
                    ss << namedAttr.getName().getValue() << " = ";
                    auto valueAttr = namedAttr.getValue();
                    return exportConstantAttr(valueAttr, ss);
                });
        }
        if (auto aggAttr = mlir::dyn_cast<P4HIR::AggAttr>(constantAttr)) {
            ss << "{";
            auto result = Utilities::interleaveCommaWithError(
                aggAttr.getFields(), ss, [&](auto elem) { return exportConstantAttr(elem, ss); });
            ss << "}";
            return result;
        }
        if (auto ctorAttr = mlir::dyn_cast<P4HIR::CtorParamAttr>(constantAttr)) {
            ss << ctorAttr.getName().strref();
            return mlir::success();
        }
        return mlir::emitError(loc)
               << "Unsupported attribute type for P4 constant: " << constantAttr;
    }

    /// @brief Exports a P4 L-Value (left-hand side of assignment or out/inout argument).
    /// @param value The MLIR Value representing the L-Value (often a VariableOp or ExtractRefOp
    /// result).
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// myVar          // from %v = p4hir.variable ...
    /// myHdr.field    // from %f = p4hir.struct_extract_ref %h["field"] ...
    /// myStack[idx]   // from %e = p4hir.array_get %s[%idx] ... (used as LVal).
    /// myVar[H:L]     // from %sl = p4hir.slice_ref %v[H:L] ...
    /// ```
    mlir::LogicalResult exportLValue(mlir::Value value, ExtendedFormattedOStream &ss) {
        mlir::Operation *op = value.getDefiningOp();
        auto loc = (op != nullptr) ? op->getLoc() : mlir::UnknownLoc::get(value.getContext());

        // L-Value can be a mutable parameter (out, inout).
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            return exportParameterReference(blockArg, ss);
        }

        if (op == nullptr) {
            return mlir::emitError(loc)
                   << "LValue has no defining operation and is not a block argument: " << value;
        }

        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            // Base case: a simple variable.
            if (auto varName = varOp.getName()) {
                ss << varName.value();
                return mlir::success();
            }
            return varOp.emitError() << "VariableOp used as LValue has no name.";
        }
        if (auto extractRefOp = mlir::dyn_cast<P4HIR::StructExtractRefOp>(op)) {
            // Field access: base.field.
            // Recursively export the base L-Value.
            if (failed(exportLValue(extractRefOp.getInput(), ss))) {
                return mlir::failure();
            }
            // Skip validity bits.
            if (extractRefOp.getFieldName() == P4HIR::HeaderType::validityBit) {
                return mlir::success();
            }
            // Normal field access.
            ss << "." << extractRefOp.getFieldName();
            return mlir::success();
        }
        if (auto arrayGetOp = mlir::dyn_cast<P4HIR::ArrayGetOp>(op)) {
            // Array element access: base[index].
            // This handles using the *result* of array_get as an LValue (if mutable).
            // We need to form the P4 `base[index]` syntax.
            if (failed(exportLValue(arrayGetOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << "[";
            if (failed(exportExpression(arrayGetOp.getIndex(), ss))) {
                return mlir::failure();
            }
            ss << "]";
            return mlir::success();
        }
        if (auto sliceRefOp = mlir::dyn_cast<P4HIR::SliceRefOp>(op)) {
            // Slice access: base[H:L].
            if (failed(exportLValue(sliceRefOp.getInput(), ss))) {
                return mlir::failure();
            }
            ss << "[" << sliceRefOp.getHighBit() << ":" << sliceRefOp.getLowBit() << "]";
            return mlir::success();
        }
        return op->emitError() << "Unsupported operation '" << op->getName() << "' for LValue use";
    }

    /// @brief Exports a reference to a function/action/parser/control parameter.
    /// @param value The MLIR BlockArgument representing the parameter.
    /// @param ss The output stream.
    /// @return Success or failure.
    static mlir::LogicalResult exportParameterReference(mlir::Value value,
                                                        ExtendedFormattedOStream &ss) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            unsigned argIndex = blockArg.getArgNumber();
            mlir::Block *ownerBlock = blockArg.getOwner();
            mlir::Operation *parentOp = ownerBlock->getParentOp();

            // Find the original parameter name stored in the function-like op attribute.
            if (auto fun = mlir::dyn_cast<mlir::FunctionOpInterface>(parentOp)) {
                auto nameAttr = fun.getArgAttr(argIndex, P4HIR::FuncOp::getParamNameAttrName());
                if (auto nameStrAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(nameAttr)) {
                    ss << nameStrAttr.getValue();
                    return mlir::success();
                }
                return fun->emitError() << "Param name attribute has unexpected type for arg "
                                        << argIndex << ": " << nameAttr;
            }
            return parentOp->emitError()
                   << "exportParameterReference called with non-FunctionOp parent: " << *parentOp;
        }
        return mlir::emitError(mlir::UnknownLoc::get(value.getContext()))
               << "exportParameterReference called with non-BlockArgument: " << value;
    }

    //===========================================================================================//
    // Expression Helper Functions.
    //===========================================================================================//

    /// @brief Gets the P4 operator string for a p4hir.unary kind.
    static mlir::FailureOr<llvm::StringRef> getUnaryOpString(P4HIR::UnaryOpKind kind) {
        switch (kind) {
            case P4HIR::UnaryOpKind::Neg:
                return llvm::StringRef("-");
            case P4HIR::UnaryOpKind::UPlus:
                return llvm::StringRef("+");
            case P4HIR::UnaryOpKind::Cmpl:
                return llvm::StringRef("~");
            case P4HIR::UnaryOpKind::LNot:
                return llvm::StringRef("!");
        }
        return mlir::failure();
    }

    /// @brief Gets the P4 operator string for a p4hir.binop kind.
    static mlir::FailureOr<llvm::StringRef> getBinaryOpString(P4HIR::BinOpKind kind) {
        switch (kind) {
            case P4HIR::BinOpKind::Mul:
                return llvm::StringRef("*");
            case P4HIR::BinOpKind::Div:
                return llvm::StringRef("/");
            case P4HIR::BinOpKind::Mod:
                return llvm::StringRef("%");
            case P4HIR::BinOpKind::Add:
                return llvm::StringRef("+");
            case P4HIR::BinOpKind::Sub:
                return llvm::StringRef("-");
            case P4HIR::BinOpKind::AddSat:
                return llvm::StringRef("|+|");
            case P4HIR::BinOpKind::SubSat:
                return llvm::StringRef("|-|");
            // Note: Bitwise OR.
            case P4HIR::BinOpKind::Or:
                return llvm::StringRef("|");
            // Note: Bitwise XOR.
            case P4HIR::BinOpKind::Xor:
                return llvm::StringRef("^");
            // Note: Bitwise AND.
            case P4HIR::BinOpKind::And:
                return llvm::StringRef("&");
        }
        return mlir::failure();
    }

    /// @brief Gets the P4 operator string for a p4hir.cmp kind.
    static mlir::FailureOr<llvm::StringRef> getCompareOpString(P4HIR::CmpOpKind kind) {
        switch (kind) {
            case P4HIR::CmpOpKind::Lt:
                return llvm::StringRef("<");
            case P4HIR::CmpOpKind::Le:
                return llvm::StringRef("<=");
            case P4HIR::CmpOpKind::Gt:
                return llvm::StringRef(">");
            case P4HIR::CmpOpKind::Ge:
                return llvm::StringRef(">=");
            case P4HIR::CmpOpKind::Eq:
                return llvm::StringRef("==");
            case P4HIR::CmpOpKind::Ne:
                return llvm::StringRef("!=");
        }
        return mlir::failure();
    }

    //===========================================================================================//
    // Statement Exporting (General).
    //===========================================================================================//

    /// @brief Exports a single P4 variable declaration statement.
    /// @param varOp The VariableOp defining the variable (used for type and name).
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// bit<32> myVariable;
    /// HeaderType myHeaderInstance;
    /// ```
    mlir::LogicalResult exportP4Declaration(P4HIR::VariableOp &varOp,
                                            ExtendedFormattedOStream &ss) {
        // The type of the *result* of VariableOp is ref<T>. We need the object type T.
        auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(varOp.getResult().getType());
        if (!refType) {
            return varOp.emitError() << "VariableOp result is not a ReferenceType.";
        }
        mlir::Type objectType = refType.getObjectType();
        if (!varOp.getName()) {
            return varOp.emitError() << "VariableOp has no name.";
        }

        // Handle array types (header stacks) specifically for size declaration.
        if (auto arrayType = mlir::dyn_cast<P4HIR::ArrayType>(objectType)) {
            if (failed(exportP4Type(arrayType.getElementType(), ss))) {
                return mlir::failure();
            }
            ss << " " << varOp.getName().value();
            ss << "[" << arrayType.getSize() << "]";
            return mlir::success();
        }
        // Normal type declaration.
        if (failed(exportP4Type(objectType, ss))) {
            return mlir::failure();
        }
        ss << " " << varOp.getName().value();
        return mlir::success();
    }

    /// @brief Exports a p4hir.assign operation as a P4 assignment statement.
    /// @param assignOp The AssignOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// myVar = expression;
    /// myHdr.field = anotherVar;
    /// myHdr.setValid();   // Special case for assigning true to validity.
    /// myHdr.setInvalid(); // Special case for assigning false to validity.
    /// ```
    mlir::LogicalResult exportAssignmentStatement(P4HIR::AssignOp assignOp,
                                                  ExtendedFormattedOStream &ss) {
        mlir::Value lvalue = assignOp.getRef();
        mlir::Value rvalue = assignOp.getValue();

        // Special case: Assigning true/false to header validity bit.
        if (auto extractRef = lvalue.getDefiningOp<P4HIR::StructExtractRefOp>()) {
            if (extractRef.getFieldName() == P4HIR::HeaderType::validityBit) {
                // Check the R-Value to determine setValid or setInvalid.
                if (auto constOp = rvalue.getDefiningOp<P4HIR::ConstOp>()) {
                    // Export the base header L-Value first.
                    if (failed(exportLValue(extractRef.getInput(), ss))) {
                        return mlir::failure();
                    }
                    if (auto boolAttr = mlir::dyn_cast<P4HIR::BoolAttr>(constOp.getValue())) {
                        // Append the correct method call.
                        ss << (boolAttr.getValue() ? ".setValid()" : ".setInvalid()");
                        ss.semicolon();
                        return mlir::success();
                    }
                    if (auto validityBitAttr =
                            mlir::dyn_cast<P4HIR::ValidityBitAttr>(constOp.getValue())) {
                        // Append the correct method call.
                        ss << (validityBitAttr.getValue() == P4HIR::ValidityBit::Valid
                                   ? ".setValid()"
                                   : ".setInvalid()");
                        ss.semicolon();
                        return mlir::success();
                    }
                    return constOp.emitError() << "Constant value is not a boolean.";
                }
                return assignOp.emitError() << "Assigning non-const to header validity "
                                               "bit might need lowering.";
            }
        }

        if (failed(exportLValue(lvalue, ss))) {
            return mlir::failure();
        }
        ss << " = ";
        if (failed(exportExpression(rvalue, ss))) {
            return mlir::failure();
        }
        ss.semicolon();
        return mlir::success();
    }

    /// @brief Exports a p4hir.assign_slice operation as a P4 slice assignment statement.
    /// @param assignOp The AssignSliceOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// myVar[7:4] = expression;
    /// ```
    mlir::LogicalResult exportAssignSliceStatement(P4HIR::AssignSliceOp assignOp,
                                                   ExtendedFormattedOStream &ss) {
        // Base variable reference.
        mlir::Value lvalue = assignOp.getRef();
        mlir::Value rvalue = assignOp.getValue();

        // Export the base L-Value (e.g., myVar).
        if (failed(exportLValue(lvalue, ss))) {
            return mlir::failure();
        }
        // Append the slice specification.
        ss << "[" << assignOp.getHighBit() << ":" << assignOp.getLowBit() << "]";

        // Assignment part.
        ss << " = ";
        if (failed(exportExpression(rvalue, ss))) {
            return mlir::failure();
        }
        ss.semicolon();
        return mlir::success();
    }

    //===========================================================================================//
    // Statement Exporting (Control Flow).
    //===========================================================================================//

    /// @brief Exports a p4hir.if operation as a P4 if-else statement.
    /// @param ifOp The IfOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// if (condition) {
    ///     // then block.
    /// } else {
    ///     // else block.
    /// }
    /// ```
    mlir::LogicalResult exportIf(P4HIR::IfOp &ifOp, ExtendedFormattedOStream &ss) {
        ss << "if (";
        if (failed(exportExpression(ifOp.getCondition(), ss))) {
            return mlir::failure();
        }
        ss << ") ";

        // Export the 'then' region.
        ss.openBrace();
        hoistedAssignOps.clear();
        if (!ifOp.getThenRegion().empty()) {
            // Export the remaining statements.
            if (failed(exportBlock(ifOp.getThenRegion().front(), ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();

        // Export the 'else' region if it exists and is not empty.
        // TODO: Assume there is always a yieldOP present. Is that accurate?
        auto &elseRegion = ifOp.getElseRegion();
        if (elseRegion.empty() || elseRegion.front().empty() ||
            elseRegion.front().getOperations().size() == 1) {
            return mlir::success();
        }

        ss << " else ";
        ss.openBrace();
        hoistedAssignOps.clear();
        // Export the remaining statements.
        if (failed(exportBlock(ifOp.getElseRegion().front(), ss))) {
            return mlir::failure();
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult exportSwitchOp(P4HIR::SwitchOp switchOp, ExtendedFormattedOStream &ss) {
        ss << "switch (";
        if (failed(exportExpression(switchOp.getCondition(), ss))) {
            return mlir::failure();
        }
        ss << ") ";
        ss.openBrace();

        if (switchOp.getBody().empty()) {
            ss.closeBrace();
            return mlir::success();
        }

        auto &bodyBlock = switchOp.getBody().front();
        for (auto caseOp : bodyBlock.getOps<P4HIR::CaseOp>()) {
            if (failed(exportSwitchCase(caseOp, ss))) {
                return mlir::failure();
            }
        }

        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult exportSwitchCase(P4HIR::CaseOp caseOp, ExtendedFormattedOStream &ss) {
        auto kind = caseOp.getKind();
        auto values = caseOp.getValueAttr();

        switch (kind) {
            case P4HIR::CaseOpKind::Equal: {
                if (values.size() != 1) {
                    return caseOp.emitError() << "p4hir.case (equal) has zero or multiple values, "
                                                 "P4 switch expects exactly one.";
                }
                auto valAttr = values[0];
                // TODO: Would be nice not to special-case this one.
                // This will also break if we are matching on an enum.
                if (auto enumAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(valAttr)) {
                    ss << enumAttr.getField().strref();
                } else {
                    if (failed(exportConstantAttr(valAttr, ss))) {
                        return mlir::failure();
                    }
                }
                ss << ": ";
                break;
            }
            case P4HIR::CaseOpKind::Anyof: {
                // P4 syntax: case VAL1: case VAL2: ... case VALN: { statements }
                for (size_t i = 0; i < values.size(); ++i) {
                    auto valAttr = values[i];
                    // TODO: Would be nice not to special-case this one.
                    // This will also break if we are matching on an enum.
                    if (auto enumAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(valAttr)) {
                        ss << enumAttr.getField().strref();
                    } else {
                        if (failed(exportConstantAttr(valAttr, ss))) {
                            return mlir::failure();
                        }
                    }
                    ss << ":";
                    if (i < values.size() - 1) {
                        ss << " ";
                    }
                }
                ss << " ";
                break;
            }
            case P4HIR::CaseOpKind::Default: {
                ss << "default: ";
                break;
            }
        }

        ss.openBrace();
        // The case region should contain a single block.
        if (!caseOp.getCaseRegion().empty()) {
            hoistedAssignOps.clear();
            if (failed(exportBlock(caseOp.getCaseRegion().front(), ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    //===========================================================================================//
    // Statement Exporting (Common Context).
    //===========================================================================================//

    /// @brief Exports statement operations common to most P4 execution blocks (actions, apply,
    /// etc.).
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or Failure state, plus a boolean indicating if the op was handled (even if
    /// skipped).
    llvm::FailureOr<bool> exportCommonStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        // Operations that primarily define values used in expressions are skipped
        // when they appear standalone, assuming their result is used later.
        if (mlir::isa<P4HIR::ReadOp, P4HIR::CastOp, P4HIR::StructExtractOp,
                      P4HIR::StructExtractRefOp, P4HIR::TupleExtractOp, P4HIR::SliceOp,
                      P4HIR::SliceRefOp, P4HIR::ConstOp, P4HIR::UnaryOp, P4HIR::BinOp, P4HIR::CmpOp,
                      P4HIR::ConcatOp, P4HIR::ShlOp, P4HIR::ShrOp, P4HIR::StructOp, P4HIR::ArrayOp,
                      P4HIR::ArrayGetOp, P4HIR::UninitializedOp, P4HIR::TernaryOp>(op)) {
            // Assume used later by exportExpression, so skip statement emission here.
            return true;
        }

        // Variable declarations are handled by hoisting, skip here.
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            // Indicate handled (by hoisting), but do nothing.
            return true;
        }

        // Handle statement-level operations.
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            // Assignments are handled normally here, *unless* they were
            // part of an initialization already hoisted (caller skips those).
            if (failed(exportAssignmentStatement(assignOp, ss))) {
                return mlir::failure();
            }
            return true;
        }
        if (auto assignSliceOp = mlir::dyn_cast<P4HIR::AssignSliceOp>(op)) {
            if (failed(exportAssignSliceStatement(assignSliceOp, ss))) {
                return mlir::failure();
            }
            return true;
        }
        if (auto returnOp = mlir::dyn_cast<P4HIR::ReturnOp>(op)) {
            ss << "return";
            if (returnOp.getNumOperands() == 1) {
                ss << " ";
                if (failed(exportExpression(returnOp.getOperand(0), ss))) {
                    return mlir::failure();
                }
            } else if (returnOp.getNumOperands() > 1) {
                return returnOp.emitError()
                       << "P4 return supports 0 or 1 operand, found " << returnOp.getNumOperands();
            }
            ss.semicolon();
            return true;
        }
        if (auto exitOp = mlir::dyn_cast<P4HIR::ExitOp>(op)) {
            ss << "exit";
            ss.semicolon();
            return true;
        }
        if (auto ifOp = mlir::dyn_cast<P4HIR::IfOp>(op)) {
            if (failed(exportIf(ifOp, ss))) {
                return mlir::failure();
            }
            // IfOp itself handles the semicolon/newline internally.
            return true;
        }
        if (auto callOp = mlir::dyn_cast<P4HIR::CallOp>(op)) {
            // Direct function/action call as a statement.
            ss << callOp.getCalleeAttr().getLeafReference().getValue() << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    callOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            ss.semicolon();
            return true;
        }
        if (auto callMethodOp = mlir::dyn_cast<P4HIR::CallMethodOp>(op)) {
            // Method call as a statement.
            if (failed(exportExpression(callMethodOp.getBase(), ss))) {
                return mlir::failure();
            }
            ss << "." << callMethodOp.getCallee().getLeafReference().getValue();
            if (auto typeOperands = callMethodOp.getTypeOperandsAttr()) {
                if (failed(exportTypeAttributes(typeOperands.getValue(), ss))) {
                    return mlir::failure();
                }
            }
            ss << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    callMethodOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            ss.semicolon();
            return true;
        }
        if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(op)) {
            // TODO: What about values that could be returned from the scope?
            ss.openBrace();
            if (!scopeOp.getScopeRegion().empty()) {
                if (failed(exportBlock(scopeOp.getScopeRegion().front(), ss, false))) {
                    return mlir::failure();
                }
            }
            ss.closeBrace();
            return true;
        }
        if (auto applyOp = mlir::dyn_cast<P4HIR::ApplyOp>(op)) {
            // parser.apply() or control.apply().
            if (failed(exportExpression(applyOp.getCallee(), ss))) {
                return mlir::failure();
            }
            ss << ".apply(";
            if (failed(Utilities::interleaveCommaWithError(
                    applyOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            ss.semicolon();
            return true;
        }
        if (auto tableApplyOp = mlir::dyn_cast<P4HIR::TableApplyOp>(op)) {
            // table.apply().
            ss << tableApplyOp.getCallee().getLeafReference().strref() << ".apply()";
            ss.semicolon();
            return true;
        }
        if (auto switchOp = mlir::dyn_cast<P4HIR::SwitchOp>(op)) {
            if (failed(exportSwitchOp(switchOp, ss))) {
                return mlir::failure();
            }
            return true;
        }
        // TODO: Add ForOp, ForInOp etc.
        // Indicate operation was not handled by this function.
        return false;
    }

    /// @brief Wrapper for exportCommonStatement that returns mlir::LogicalResult.
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or Failure.
    mlir::LogicalResult exportCommonStatementWithError(mlir::Operation &op,
                                                       ExtendedFormattedOStream &ss) {
        auto result = exportCommonStatement(op, ss);
        if (failed(result)) {
            return mlir::failure();
        }
        if (!result.value()) {
            // If common statement didn't handle it, emit error.
            return op.emitError() << "Unsupported operation '" << op.getName()
                                  << "' in common statement context: ";
        }
        return mlir::success();
    }

    //===========================================================================================//
    // Statement Exporting (Control/Table Context).
    //===========================================================================================//

    /// @brief Exports a P4 table definition.
    /// @param tableOp The TableOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportTableDeclaration(P4HIR::TableOp &tableOp,
                                               ExtendedFormattedOStream &ss) {
        ss << "table " << tableOp.getName();
        ss << " ";
        ss.openBrace();

        // 1. Export Key (finds TableKeyOp and its contents).
        if (failed(exportTableKeyProperty(tableOp, ss))) {
            return mlir::failure();
        }

        // 2. Export Actions (finds TableActionsOp and its contents).
        if (failed(exportTableActionsProperty(tableOp, ss))) {
            return mlir::failure();
        }

        // 3. Export Default Action (finds TableDefaultActionOp).
        if (failed(exportTableDefaultActionProperty(tableOp, ss))) {
            return mlir::failure();
        }

        // 4. Export Size (finds TableSizeOp).
        if (failed(exportTableSizeProperty(tableOp, ss))) {
            return mlir::failure();
        }

        // 5. Export Generic Properties (finds TableEntryOp instances).
        //    This implicitly handles 'implementation' if it's stored as TableEntryOp.
        if (failed(exportTableGenericProperties(tableOp, ss))) {
            return mlir::failure();
        }

        // TODO: Const entries.
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports the 'key' property of a table.
    ///        Finds the TableKeyOp inside the TableOp's body.
    mlir::LogicalResult exportTableKeyProperty(P4HIR::TableOp &tableOp,
                                               ExtendedFormattedOStream &ss) {
        P4HIR::TableKeyOp keyOp;
        for (auto &op : tableOp.getBody().front()) {
            if (auto foundKeyOp = mlir::dyn_cast<P4HIR::TableKeyOp>(op)) {
                keyOp = foundKeyOp;
                break;
            }
        }

        if (!keyOp) {
            // Table does not have a key.
            return mlir::success();
        }

        // Export annotations for the key block itself (if any on TableKeyOp).
        if (auto annots = getAnnotationsAttr(keyOp)) {
            if (failed(exportAnnotations(*annots, ss))) {
                return mlir::failure();
            }
        }

        ss << "key = ";
        ss.openBrace();
        // Iterate through ops *inside* the TableKeyOp's region.
        if (!keyOp.getBody().empty()) {
            for (auto &entryOp : keyOp.getBody().front()) {
                if (auto keyEntryOp = mlir::dyn_cast<P4HIR::TableKeyEntryOp>(entryOp)) {
                    // Export the expression being matched.
                    if (failed(exportExpression(keyEntryOp.getValue(), ss))) {
                        return mlir::failure();
                    }
                    ss << " : ";

                    // Export the match kind.
                    ss << keyEntryOp.getMatchKind().getValue().getValue();
                    ss << " ";
                    // Export annotations for the specific key entry.
                    if (auto entryAnnots = getAnnotationsAttr(keyEntryOp)) {
                        if (failed(exportAnnotations(*entryAnnots, ss))) {
                            return mlir::failure();
                        }
                    }
                    ss.semicolon();
                } else {
                    // TODO: What to do here?.
                    // return entryOp.emitError() << "Unexpected operation inside p4hir.table_key
                    // region";
                }
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports the 'actions' property of a table.
    ///        Finds the TableActionsOp inside the TableOp's body.
    mlir::LogicalResult exportTableActionsProperty(P4HIR::TableOp &tableOp,
                                                   ExtendedFormattedOStream &ss) {
        P4HIR::TableActionsOp actionsOp;
        for (auto &op : tableOp.getBody().front()) {
            if (auto foundActionsOp = mlir::dyn_cast<P4HIR::TableActionsOp>(op)) {
                actionsOp = foundActionsOp;
                break;
            }
        }

        if (!actionsOp) {
            return mlir::success();
        }

        // Export annotations for the actions block itself (if any on TableActionsOp).
        if (auto annots = getAnnotationsAttr(actionsOp)) {
            if (failed(exportAnnotations(*annots, ss))) {
                return mlir::failure();
            }
        }

        if (actionsOp.getBody().empty()) {
            ss << "actions = {}";
            return mlir::success();
        }
        // Iterate through ops *inside* the TableActionsOp's region.
        ss << "actions = ";
        ss.openBrace();
        for (auto &entryOp : actionsOp.getBody().front()) {
            if (auto actionEntryOp = mlir::dyn_cast<P4HIR::TableActionOp>(entryOp)) {
                if (auto entryAnnots = getAnnotationsAttr(actionEntryOp)) {
                    if (failed(exportAnnotations(*entryAnnots, ss))) {
                        return mlir::failure();
                    }
                }

                auto &callBody = actionEntryOp.getBody();
                bool printedCall = false;
                // Check if the region contains a specific call generated by visit(expr).
                for (auto &block : callBody) {
                    for (auto &op : block) {
                        if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(op)) {
                            return scopeOp.emitError()
                                   << "Method calls are not supported yet in table action lists.";
                            printedCall = true;
                            break;
                        }
                    }
                }

                // If no specific call was found/printed from the region:.
                if (!printedCall) {
                    // Print the action name referenced by the TableActionOp attribute.
                    ss << actionEntryOp.getActionAttr().getLeafReference().getValue();
                }

                ss.semicolon();
            } else {
                // Unexpected op inside actions region.
                entryOp.emitError() << "Unexpected operation inside p4hir.table_actions region.";
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports the 'default_action' property of a table.
    ///        Finds the TableDefaultActionOp inside the TableOp's body.
    mlir::LogicalResult exportTableDefaultActionProperty(P4HIR::TableOp &tableOp,
                                                         ExtendedFormattedOStream &ss) {
        P4HIR::TableDefaultActionOp defaultActionOp;
        for (auto &op : tableOp.getBody().front()) {
            if (auto foundOp = mlir::dyn_cast<P4HIR::TableDefaultActionOp>(op)) {
                defaultActionOp = foundOp;
                break;
            }
        }

        if (!defaultActionOp) {
            // Default action is optional.
            return mlir::success();
        }

        // Export annotations for the default action assignment.
        if (auto annots = getAnnotationsAttr(defaultActionOp)) {
            if (failed(exportAnnotations(*annots, ss))) {
                return mlir::failure();
            }
        }
        // TODO: This might not always be const.
        ss << "default_action = ";

        // The value is inside the region, likely yielded or represented by a call op.
        if (defaultActionOp.getBody().empty() || defaultActionOp.getBody().front().empty()) {
            return defaultActionOp.emitError() << "p4hir.table_default_action has empty region.";
        }
        // Try to get the right op...
        // TODO: Figure out a simpler way to get this. Should likely be a single call op inside.
        for (auto &op : defaultActionOp.getOps()) {
            if (failed(exportExpression(op, ss))) {
                return op.emitError() << "Cannot determine default action value expression.";
            }
            ss.semicolon();
            return mlir::success();
        }
        return defaultActionOp.emitError() << "Could not find default action value expression.";
    }

    /// @brief Exports the 'size' property of a table.
    ///        Finds the TableSizeOp inside the TableOp's body.
    mlir::LogicalResult exportTableSizeProperty(P4HIR::TableOp &tableOp,
                                                ExtendedFormattedOStream &ss) {
        P4HIR::TableSizeOp sizeOp;
        for (auto &op : tableOp.getBody().front()) {
            if (auto foundOp = mlir::dyn_cast<P4HIR::TableSizeOp>(op)) {
                sizeOp = foundOp;
                break;
            }
        }

        if (!sizeOp) {
            // Size is optional.
            return mlir::success();
        }

        // Export annotations for the size property.
        if (auto annots = getAnnotationsAttr(sizeOp)) {
            if (failed(exportAnnotations(*annots, ss))) {
                return mlir::failure();
            }
        }

        ss << "const size = ";
        auto sizeAttr = sizeOp.getValue();
        if (failed(exportConstantAttr(sizeAttr, ss))) {
            return mlir::failure();
        }

        ss.semicolon();
        return mlir::success();
    }

    /// @brief Exports generic table properties represented by TableEntryOp.
    ///        Includes special handling for 'implementation'.
    mlir::LogicalResult exportTableGenericProperties(P4HIR::TableOp &tableOp,
                                                     ExtendedFormattedOStream &ss) {
        for (auto &op : tableOp.getBody().front()) {
            if (auto entryOp = mlir::dyn_cast<P4HIR::TableEntryOp>(op)) {
                // Export annotations for the property assignment.
                if (auto annots = getAnnotationsAttr(entryOp)) {
                    if (failed(exportAnnotations(*annots, ss))) {
                        return mlir::failure();
                    }
                }

                // Get property name.
                std::string propName = entryOp.getName().lower();
                bool isConstant = entryOp.getIsConst();

                if (isConstant) {
                    ss << "const ";
                }
                ss << propName << " = ";

                // Value is yielded from the region.
                if (entryOp.getBody().empty() || entryOp.getBody().front().empty()) {
                    return entryOp.emitError() << "TableEntryOp region must yield a value.";
                }
                auto yieldOp = mlir::dyn_cast_if_present<P4HIR::YieldOp>(
                    entryOp.getBody().front().getTerminator());
                if (yieldOp.getNumOperands() != 1) {
                    return yieldOp.emitError() << "TableEntryOp yield must have one operand.";
                }

                // Export the yielded value using exportExpression.
                if (failed(exportExpression(yieldOp.getOperand(0), ss))) {
                    return mlir::failure();
                }
                ss.semicolon();
            }
        }
        return mlir::success();
    }

    /// @brief Exports a statement found directly within a control definition (outside apply).
    ///        This typically includes local variables, actions, and tables.
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportControlStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        // Variable declarations are handled before via hoisting.
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            return mlir::success();
        }
        // Handled assignments were also dealt with by the caller.
        if (hoistedAssignOps.count(&op) != 0U) {
            return mlir::success();
        }

        // Action Definitions.
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op)) {
            if (funcOp.getAction()) {
                // Re-use function export logic.
                return exportTopLevelFuncOp(funcOp, ss);
            }
            // Non-action functions usually not defined directly in control scope.
            return op.emitError() << "Unexpected FuncOp (non-action) in control scope.";
        }
        // Table Definitions.
        if (auto tableOp = mlir::dyn_cast<P4HIR::TableOp>(op)) {
            return exportTableDeclaration(tableOp, ss);
        }

        // Instantiations (e.g., counters, meters, registers, sub-controls).
        if (auto instOp = mlir::dyn_cast<P4HIR::InstantiateOp>(op)) {
            // Export annotations associated with the instance declaration.
            if (auto annots = getAnnotationsAttr(instOp)) {
                if (failed(exportAnnotations(*annots, ss))) {
                    return mlir::failure();
                }
            }

            // Export the type being instantiated (e.g., "MyCounter<bit<32>>").
            mlir::Type instanceType = instOp.getResult().getType();
            if (failed(exportP4Type(instanceType, ss))) {
                return mlir::failure();
            }

            // Export the constructor arguments (e.g., "(1024)").
            ss << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    instOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";

            // Export the instance name (e.g., " my_counter_instance").
            ss << " " << instOp.getName();
            ss.semicolon();
            return mlir::success();
        }

        // Apply Block.
        if (auto controlApplyOp = mlir::dyn_cast<P4HIR::ControlApplyOp>(op)) {
            return exportControlApply(controlApplyOp, ss);
        }

        // Try common statements if none of the above match.
        auto commonResult = exportCommonStatement(op, ss);
        if (failed(commonResult)) {
            return mlir::failure();
        }
        if (commonResult.value()) {
            return mlir::success();
        }

        return op.emitError() << " Unsupported control statement '" << op.getName() << "'";
    }

    /// @brief Exports the apply block of a control.
    /// @param op The ControlApplyOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// apply {
    ///     // Variable Declarations.
    ///     bit<1> temp_flag = false;
    ///
    ///     // statements...
    ///     my_table.apply();
    /// }
    /// ```
    mlir::LogicalResult exportControlApply(P4HIR::ControlApplyOp &op,
                                           ExtendedFormattedOStream &ss) {
        ss << "apply ";
        if (op.getBody().empty()) {
            ss << "{}";
            return mlir::success();
        }
        ss.openBrace();
        hoistedAssignOps.clear();
        // Export the remaining statements.
        if (failed(exportBlock(op.getBody().front(), ss))) {
            return mlir::failure();
        }
        ss.closeBrace();
        return mlir::success();
    }

    //===========================================================================================//
    // Statement Exporting (Parser Context).
    //===========================================================================================//

    /// @brief Exports a statement found directly within a parser definition (outside states).
    ///        This typically includes local variables and state definitions.
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportParserStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        // Variable declarations are handled before via hoisting.
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            return mlir::success();
        }
        if (hoistedAssignOps.count(&op) != 0U) {
            return mlir::success();
        }

        // State Definitions.
        if (auto parserState = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
            return exportParserState(parserState, ss);
        }
        if (mlir::isa<P4HIR::ParserTransitionOp, P4HIR::ParserAcceptOp, P4HIR::ParserRejectOp>(
                op)) {
            // These are currently noops.
            return mlir::success();
        }

        // Instantiations (if allowed at parser scope).
        if (auto instOp = mlir::dyn_cast<P4HIR::InstantiateOp>(op)) {
            // Export annotations associated with the instance declaration.
            if (auto annots = getAnnotationsAttr(instOp)) {
                if (failed(exportAnnotations(*annots, ss))) {
                    return mlir::failure();
                }
            }
            mlir::Type instanceType = instOp.getResult().getType();
            if (failed(exportP4Type(instanceType, ss))) {
                return mlir::failure();
            }
            ss << "(";
            if (failed(Utilities::interleaveCommaWithError(
                    instOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg, ss); }))) {
                return mlir::failure();
            }
            ss << ")";
            ss << " " << instOp.getName();
            ss.semicolon();
            return mlir::success();
        }

        // Try common statements if none of the above match.
        auto commonResult = exportCommonStatement(op, ss);
        if (failed(commonResult)) {
            return mlir::failure();
        }
        if (commonResult.value()) {
            return mlir::success();
        }

        // Default error for Unsupported parser-level ops.
        return op.emitError() << " Unsupported parser statement '" << op.getName() << "'";
    }

    /// @brief Exports a parser state definition and its contents.
    /// @param parserState The ParserStateOp to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// state parse_ipv4 {
    ///     // Variable Declarations.
    ///     bit<16> next_proto;
    ///
    ///     // State logic...
    ///     pkt.extract(hdr.ipv4);
    ///     next_proto = hdr.ipv4.protocol;
    ///     transition select(next_proto) { /* ... */ }
    /// }
    /// ```
    mlir::LogicalResult exportParserState(P4HIR::ParserStateOp &parserState,
                                          ExtendedFormattedOStream &ss) {
        auto parserStateName = parserState.getName();
        if (parserStateName == "accept" || parserStateName == "reject") {
            // Ignore accept or reject state.
            return mlir::success();
        }
        ss << "state " << parserStateName << " ";
        ss.openBrace();
        // Export statements within the state's block.
        if (!parserState.getBody().empty()) {
            // Export the remaining state logic statements.
            if (failed(exportBlock(parserState.getBody().front(), ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    /// @brief Exports a statement found within a parser state.
    /// @param op The MLIR operation to export.
    /// @param ss The output stream.
    /// @return Success or failure.
    mlir::LogicalResult exportParserStateStatement(mlir::Operation &op,
                                                   ExtendedFormattedOStream &ss) {
        // Variable declarations are handled before via hoisting.
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            return mlir::success();
        }
        if (hoistedAssignOps.count(&op) != 0U) {
            return mlir::success();
        }

        if (auto transitionOp = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            ss << "transition " << transitionOp.getState().getLeafReference().strref();
            ss.semicolon();
            return mlir::success();
        }
        // Accept and reject states do not have content.
        if (mlir::isa<P4HIR::ParserAcceptOp, P4HIR::ParserRejectOp>(op)) {
            return mlir::success();
        }
        // TODO: Handle 'transition select(...) { ... }' (ParserSelectOp).

        auto commonResult = exportCommonStatement(op, ss);
        if (failed(commonResult)) {
            return mlir::failure();
        }
        if (commonResult.value()) {
            return mlir::success();
        }

        return op.emitError() << " Unsupported parser state statement '" << op.getName() << "'";
    }

    //===========================================================================================//
    // Parameter List Exporting.
    //===========================================================================================//

    /// @brief Exports the full parameter list for a function, action, parser, or control.
    /// @param functionInterface The operation with parameters (must implement FunctionOpInterface).
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// (in bit<32> data, out header hdr)
    /// ```
    mlir::LogicalResult exportParameterList(mlir::FunctionOpInterface functionInterface,
                                            ExtendedFormattedOStream &ss) {
        return Utilities::interleaveCommaWithError(
            // Need to iterate using index for getArgAttr.
            llvm::seq<unsigned>(0, functionInterface.getNumArguments()), ss,
            [&](size_t index) { return exportArgument(functionInterface, ss, index); });
    }

    /// @brief Exports a single parameter declaration within a parameter list.
    /// @param functionInterface The parent operation implementing FunctionOpInterface.
    /// @param ss The output stream.
    /// @param index The index of the parameter to export.
    /// @return Success or failure.
    /// @example P4 Code (single parameter part).
    /// ```p4
    /// in bit<32> data
    /// ```
    mlir::LogicalResult exportArgument(mlir::FunctionOpInterface functionInterface,
                                       ExtendedFormattedOStream &ss, int index) {
        // Export direction (in, out, inout).
        auto directionAttr =
            functionInterface.getArgAttr(index, P4HIR::FuncOp::getDirectionAttrName());

        P4HIR::ParamDirection direction = P4HIR::ParamDirection::None;
        if (directionAttr) {
            if (auto castDir = mlir::dyn_cast<P4HIR::ParamDirectionAttr>(directionAttr)) {
                direction = castDir.getValue();
            } else {
                return functionInterface->emitError()
                       << "Invalid direction attribute type for arg " << index;
            }
        }
        if (failed(exportParameterDirection(direction, ss))) {
            return mlir::failure();
        }

        // Export type and name using helper.
        if (failed(getOriginalParamName(functionInterface, index, ss))) {
            return mlir::failure();
        }

        return mlir::success();
    }

    /// @brief Exports the P4 direction keyword for a parameter.
    /// @param direction The parameter direction enum value.
    /// @param ss The output stream.
    /// @return Success or failure.
    /// @example P4 Code.
    /// ```p4
    /// in
    /// out
    /// inout
    /// // (or nothing for None direction).
    /// ```
    static mlir::LogicalResult exportParameterDirection(P4HIR::ParamDirection direction,
                                                        ExtendedFormattedOStream &ss) {
        switch (direction) {
            case P4HIR::ParamDirection::None:
                // No keyword for default direction.
                break;
            case P4HIR::ParamDirection::In:
            case P4HIR::ParamDirection::Out:
            case P4HIR::ParamDirection::InOut:
                ss << P4HIR::stringifyEnum(direction) << " ";
                break;
        }
        return mlir::success();
    }

    /// @brief Gets the original parameter name and exports type and name.
    /// @param funcOp The parent function-like operation.
    /// @param index The index of the parameter.
    /// @param ss The output stream where Type and Name are written.
    /// @return Success or failure.
    mlir::LogicalResult getOriginalParamName(mlir::FunctionOpInterface funcOp, size_t index,
                                             ExtendedFormattedOStream &ss) {
        auto argAttrs = funcOp.getArgAttrsAttr();
        auto typeAttrs = funcOp.getArgumentTypes();
        auto paramNameAttrName = P4HIR::FuncOp::getParamNameAttrName();

        if (!argAttrs || index >= argAttrs.size()) {
            funcOp->emitError() << "Invalid arg index " << index;
        }
        auto dict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(argAttrs[index]);
        if (!dict) {
            return funcOp->emitError() << "Argument attribute is not a dictionary " << index;
        }
        auto nameAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(dict.get(paramNameAttrName));
        if (!nameAttr) {
            return funcOp->emitError() << "Parameter attribute is not a StringAttr " << index;
        }
        // Export type if name attribute was found.
        if (failed(exportP4Type(typeAttrs[index], ss))) {
            return mlir::failure();
        }
        ss << " " << nameAttr.getValue();
        return mlir::success();
    }
};

}  // namespace

//================================================================================================//
// Public API Functions
//================================================================================================//

llvm::FailureOr<std::string> exportP4HirToP4(mlir::ModuleOp module,
                                             P4HirToP4ExporterOptions options) {
    P4HirToP4Exporter exporter(options);
    std::string outputString;
    llvm::raw_string_ostream rso(outputString);
    ExtendedFormattedOStream ss(rso, options.style, options.indentLevel);
    if (failed(exporter.writeProgram(module, ss))) {
        return mlir::failure();
    }
    return rso.str();
}

mlir::LogicalResult exportP4HirToP4(mlir::ModuleOp module, llvm::raw_ostream &os,
                                    P4HirToP4ExporterOptions options) {
    P4HirToP4Exporter exporter(options);
    ExtendedFormattedOStream ss(os, options.style, options.indentLevel);
    if (failed(exporter.writeProgram(module, ss))) {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult writeP4HirToP4File(mlir::ModuleOp module,
                                       const std::filesystem::path &p4OutputFile) {
    try {
        if (p4OutputFile.has_parent_path()) {
            std::filesystem::create_directories(p4OutputFile.parent_path());
        }
        std::error_code ec;
        llvm::raw_fd_ostream outfile(p4OutputFile.c_str(), ec);
        if (ec) {
            return module.emitError() << "Failed to open file for writing '"
                                      << p4OutputFile.string() << "': " << ec.message();
        }

        if (failed(P4::P4MLIR::Utilities::exportP4HirToP4(module, outfile, {}))) {
            outfile.close();
            return mlir::failure();
        }

        if (outfile.has_error()) {
            outfile.close();
            return module.emitError() << "Failed to write to file: " + p4OutputFile.string();
        }
        outfile.close();
    } catch (const std::filesystem::filesystem_error &ex) {
        return module.emitError() << "Filesystem error processing '" << p4OutputFile.string()
                                  << "': " << ex.what();
    } catch (const std::exception &ex) {
        return module.emitError() << "Standard exception processing '" << p4OutputFile.string()
                                  << "': " << ex.what();
    } catch (...) {
        return module.emitError() << "Unknown exception occurred processing '"
                                  << p4OutputFile.string() << "'.";
    }
    return mlir::success();
}

}  // namespace P4::P4MLIR::Utilities
