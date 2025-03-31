#include "export_to_p4.h"

#include <filesystem>
#include <string>
#include <system_error>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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

// Function to find all unique types used anywhere within a ModuleOp
llvm::SetVector<mlir::Type> getAllUsedTypes(mlir::ModuleOp moduleOp) {
    llvm::SetVector<mlir::Type> collectedTypes;

    // Walk through all operations in the module
    moduleOp.walk([&](mlir::Operation *op) {
        // Collect types from operation results
        for (mlir::Value result : op->getResults()) {
            collectedTypes.insert(result.getType());
        }

        // Collect types from operation operands
        for (mlir::Value operand : op->getOperands()) {
            collectedTypes.insert(operand.getType());
        }

        // Collect types from block arguments within the operation's regions
        for (mlir::Region &region : op->getRegions()) {
            for (mlir::Block &block : region) {
                for (mlir::BlockArgument arg : block.getArguments()) {
                    collectedTypes.insert(arg.getType());
                }
            }
        }
    });

    return collectedTypes;
}

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

/**
 * @brief Finds the definition operation (PackageOp, ControlOp, ParserOp, ExternOp)
 *        that a given InstantiateOp refers to via its 'callee' symbol.
 *
 * @param instantiateOp The p4hir.instantiate operation.
 * @return Pointer to the definition operation if found, nullptr otherwise.
 */
llvm::FailureOr<mlir::Operation *> findInstantiatedOpDefinition(
    P4HIR::InstantiateOp instantiateOp) {
    if (!instantiateOp) {
        return instantiateOp.emitError() << "Provided InstantiateOp is null.";
    }

    mlir::SymbolRefAttr calleeAttr = instantiateOp.getCalleeAttr();
    if (!calleeAttr) {
        // This should ideally not happen for a valid InstantiateOp.
        return instantiateOp.emitError() << "InstantiateOp does not have a 'callee' SymbolRefAttr.";
    }

    mlir::Operation *symbolTableOp = instantiateOp->getParentOfType<mlir::ModuleOp>();
    if (symbolTableOp == nullptr) {
        return instantiateOp.emitError()
               << "Error: Could not find a parent ModuleOp (or other SymbolTable) "
               << "for the InstantiateOp.";
    }

    mlir::SymbolTableCollection symbolTable;
    mlir::Operation *definitionOp = symbolTable.lookupSymbolIn(symbolTableOp, calleeAttr);

    if (definitionOp == nullptr) {
        return instantiateOp.emitError()
               << "Definition for symbol '" << calleeAttr.getLeafReference()
               << "' referenced by InstantiateOp not found in the scope of "
               << symbolTableOp->getName() << ".";
    }
    return definitionOp;
}

/// TODO: Keep this class private until its API is finalized.
class P4HirToP4Exporter {
 public:
    mlir::LogicalResult declareTopLevelTypes(mlir::Type &type, ExtendedFormattedOStream &ss) {
        if (auto structType = llvm::dyn_cast<P4HIR::StructLikeTypeInterface>(type)) {
            for (auto field : structType.getFields()) {
                if (field.type.isa<P4HIR::StructLikeTypeInterface>()) {
                    if (failed(declareType(field.type, ss))) {
                        return mlir::failure();
                    }
                }
            }
            if (failed(declareType(structType, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto refType = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
            auto subType = refType.getObjectType();
            if (subType.isa<P4HIR::StructLikeTypeInterface>()) {
                if (failed(declareType(subType, ss))) {
                    return mlir::failure();
                }
                return mlir::success();
            }
            if (mlir::isa<P4HIR::BitsType>(subType) || mlir::isa<P4HIR::BoolType>(subType)) {
                return mlir::success();
            }
            mlir::Location loc = mlir::UnknownLoc::get(subType.getContext());
            return mlir::emitError(loc)
                   << "Unsupported P4HIR reference type for P4 top-level declaration: " << subType;
        }
        if (auto externType = llvm::dyn_cast<P4HIR::ExternType>(type)) {
            return mlir::success();
        }
        if (auto parserType = llvm::dyn_cast<P4HIR::ParserType>(type)) {
            ss << parserType.getName();
            if (failed(exportTypeArguments(parserType.getTypeArguments(), ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto controlType = llvm::dyn_cast<P4HIR::ControlType>(type)) {
            return mlir::success();
        }
        if (auto packageType = llvm::dyn_cast<P4HIR::PackageType>(type)) {
            return mlir::success();
        }
        if (mlir::isa<P4HIR::BitsType>(type) || mlir::isa<P4HIR::BoolType>(type)) {
            return mlir::success();
        }
        mlir::Location loc = mlir::UnknownLoc::get(type.getContext());
        return mlir::emitError(loc)
               << "Unsupported P4HIR type for P4 top-level declaration: " << type;
    }

    mlir::LogicalResult writeProgram(mlir::ModuleOp module, ExtendedFormattedOStream &ss) {
        // 1. Try to find all the used types, and declare all complex types.
        llvm::SetVector<mlir::Type> allTypes = getAllUsedTypes(module);
        for (mlir::Type type : allTypes) {
            if (failed(declareTopLevelTypes(type, ss))) {
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

    explicit P4HirToP4Exporter(P4HirToP4ExporterOptions options) : options_(options) {}

    [[nodiscard]] P4HirToP4ExporterOptions options() const { return options_; }

 private:
    P4HirToP4ExporterOptions options_;

    mlir::LogicalResult exportP4Type(mlir::Type type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::ReferenceType>([&](P4HIR::ReferenceType ref) {
                // Recursively get the underlying object type's name
                return exportP4Type(ref.getObjectType(), ss);
            })
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
            .Case<P4HIR::ExternType>([&](P4HIR::ExternType externType) {
                ss << externType.getName();
                return mlir::success();
            })
            .Case<P4HIR::BitsType>([&](P4HIR::BitsType bitType) {
                ss << "bits<" << bitType.getWidth() << ">";
                return mlir::success();
            })
            .Case<P4HIR::BoolType>([&](P4HIR::BoolType /*boolType*/) {
                ss << P4HIR::BoolType::getMnemonic();
                return mlir::success();
            })
            .Case<P4HIR::StringType>([&](P4HIR::StringType /*stringType*/) {
                ss << P4HIR::StringType::getMnemonic();
                return mlir::success();
            })
            .Case<P4HIR::EnumType>([&](P4HIR::EnumType /*enumType*/) {
                ss << P4HIR::EnumType::name;
                return mlir::success();
            })
            .Case<P4HIR::ErrorType>([&](P4HIR::ErrorType /*errorType*/) {
                ss << P4HIR::ErrorType::name;
                return mlir::success();
            })
            .Case<P4HIR::TypeVarType>([&](P4HIR::TypeVarType typeVarType) {
                ss << typeVarType.getName();
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
            .Default([&](mlir::Type t) {
                mlir::Location loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Unsupported P4HIR type for P4 export: " << t;
            });
    }

    mlir::LogicalResult declareStructTypeFields(llvm::ArrayRef<P4HIR::FieldInfo> fields,
                                                ExtendedFormattedOStream &ss) {
        for (auto field : fields) {
            if (field.type.isa<P4HIR::ValidBitType>()) {
                // The valid bit is implicit when declaring a P4 Structlike.
                continue;
            }
            if (failed(exportP4Type(field.type, ss))) {
                return mlir::failure();
            }
            ss << " " << field.name.strref() << ";";
            ss.newline();
        }
        return mlir::success();
    }

    mlir::LogicalResult declareType(mlir::Type &type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::HeaderType>([&](P4HIR::HeaderType headerType) {
                ss << P4HIR::HeaderType::getMnemonic() << " " << headerType.getName() << " ";
                ss.openBrace();
                if (failed(declareStructTypeFields(headerType.getFields(), ss))) {
                    return mlir::failure();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Case<P4HIR::StructType>([&](P4HIR::StructType structType) {
                ss << P4HIR::StructType::getMnemonic() << " " << structType.getName() << " ";
                ss.openBrace();
                if (failed(declareStructTypeFields(structType.getFields(), ss))) {
                    return mlir::failure();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Case<P4HIR::HeaderUnionType>([&](P4HIR::HeaderUnionType unionType) {
                ss << P4HIR::HeaderUnionType::getMnemonic() << " " << unionType.getName() << " ";
                ss.openBrace();
                if (failed(declareStructTypeFields(unionType.getFields(), ss))) {
                    return mlir::failure();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                mlir::Location loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Type declaration unsupported for: " << t;
            });
    }

    mlir::LogicalResult exportLValue(mlir::Operation &type, ExtendedFormattedOStream &ss) {
        if (auto var = mlir::dyn_cast<P4HIR::StructExtractRefOp>(type)) {
            if (auto *ref = var.getInput().getDefiningOp()) {
                if (failed(exportLValue(*ref, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(exportParameterReference(var.getInput(), ss))) {
                    return mlir::failure();
                }
            }
            ss << "." << var.getFieldName();
            return mlir::success();
        }
        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(type)) {
            if (auto varName = var.getName()) {
                ss << varName.value();
            } else {
                return var.emitError() << "Variable declaration without name?";
            }
            return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(type.getContext()))
               << "Unsupported lvalue " << type;
    }

    mlir::LogicalResult exportP4Declaration(mlir::Operation &type, ExtendedFormattedOStream &ss) {
        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(type)) {
            if (failed(exportP4Type(var.getType(), ss))) {
                return mlir::failure();
            }
            if (failed(exportLValue(type, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(type.getContext()))
               << "Unsupported declaration " << type;
    }

    static mlir::LogicalResult exportConstant(P4HIR::ConstOp &constant,
                                              ExtendedFormattedOStream &ss) {
        auto attrValue = constant.getValueAttr();
        mlir::Location loc = constant.getLoc();

        if (auto intAttr = attrValue.dyn_cast<P4HIR::IntAttr>()) {
            auto bitsType = intAttr.getType().dyn_cast<P4HIR::BitsType>();
            if (!bitsType) {
                return mlir::emitError(loc)
                       << "P4HIR::IntAttr requires P4HIR::BitsType, found " << intAttr.getType();
            }
            unsigned width = bitsType.getWidth();
            const llvm::APInt &value = intAttr.getValue();
            /// TODO: Make sure we actually do not need this.
            // ss << (bitsType.isSigned() && value.isNegative() ? "-" : "") ;
            ss << width << (bitsType.isSigned() ? "s" : "w");

            std::string decimalString;
            llvm::raw_string_ostream decimalStream(decimalString);
            value.print(decimalStream, false);
            ss << decimalStream.str();
            return mlir::success();
        }
        if (auto strAttr = attrValue.dyn_cast<mlir::StringAttr>()) {
            ss << escapeP4String(strAttr.getValue());
            return mlir::success();
        }
        if (auto boolAttr = attrValue.dyn_cast<mlir::BoolAttr>()) {
            ss << (boolAttr.getValue() ? "true" : "false");
            return mlir::success();
        }

        return mlir::emitError(loc) << "Unsupported attribute type for P4 constant: " << attrValue;
    }

    mlir::LogicalResult exportConstantDeclaration(P4HIR::ConstOp &constant,
                                                  ExtendedFormattedOStream &ss) {
        ss << "const ";
        if (failed(exportP4Type(constant.getType(), ss))) {
            return mlir::failure();
        }
        ss << constant.getName() << " = ";
        if (failed(exportConstant(constant, ss))) {
            return mlir::failure();
        }
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportExpression(mlir::Operation *op, ExtendedFormattedOStream &ss) {
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            return exportConstant(constOp, ss);
        }
        if (auto cast = mlir::dyn_cast<P4HIR::CastOp>(op)) {
            ss << "(";
            if (failed(exportP4Type(cast.getType(), ss))) {
                return mlir::failure();
            }
            ss << ") ";
            if (failed(exportExpression(cast.getSrc().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            ss << var.getName();
            return mlir::success();
        }

        return op->emitError() << "Unhandled expression: " << *op;
    }

    static mlir::LogicalResult exportParameterDirection(P4HIR::ParamDirection direction,
                                                        ExtendedFormattedOStream &ss) {
        switch (direction) {
            case P4HIR::ParamDirection::None:
                return mlir::success();
            default:
                ss << P4HIR::stringifyEnum(direction) << " ";
                return mlir::success();
        }
    }

    mlir::LogicalResult exportArgument(mlir::FunctionOpInterface functionInterface,
                                       ExtendedFormattedOStream &ss, int index) {
        auto argument = functionInterface.getArgument(index);
        auto direction = functionInterface.getArgAttr(index, P4HIR::FuncOp::getDirectionAttrName());
        auto castDirection = mlir::dyn_cast<P4HIR::ParamDirectionAttr>(direction);
        if (failed(exportParameterDirection(castDirection.getValue(), ss))) {
            return mlir::failure();
        }
        if (failed(exportP4Type(argument.getType(), ss))) {
            return mlir::failure();
        }

        ss << " ";

        auto nameAttr = functionInterface.getArgAttr(index, P4HIR::FuncOp::getParamNameAttrName());
        if (nameAttr) {
            if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                ss << nameStrAttr.getValue();
            } else {
                return functionInterface->emitError()
                       << "Param name attribute has unexpected type: " << nameAttr;
            }
        } else {
            return functionInterface->emitError()
                   << "Missing param name attribute for function arg " << index;
        }
        return mlir::success();
    }

    mlir::LogicalResult exportParameterList(mlir::FunctionOpInterface functionInterface,
                                            ExtendedFormattedOStream &ss) {
        return Utilities::interleaveCommaWithError(
            functionInterface.getArguments(), ss, [&](const auto &arg) {
                return exportArgument(functionInterface, ss, arg.getArgNumber());
            });
    }

    mlir::LogicalResult exportAssignmentStatement(P4HIR::AssignOp assignOp,
                                                  ExtendedFormattedOStream &ss) {
        if (failed(exportLValue(*assignOp.getRef().getDefiningOp(), ss))) {
            return mlir::failure();
        }
        ss << " = ";
        auto value = assignOp.getValue();
        if (auto *subExpr = value.getDefiningOp()) {
            if (failed(exportExpression(subExpr, ss))) {
                return mlir::failure();
            }
        } else {
            if (failed(exportParameterReference(value, ss))) {
                return mlir::failure();
            }
        }
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportAssigningDeclaration(P4HIR::AssignOp assignOp,
                                                   ExtendedFormattedOStream &ss) {
        if (failed(exportP4Declaration(*assignOp.getRef().getDefiningOp(), ss))) {
            return mlir::failure();
        }
        ss << " = ";
        auto value = assignOp.getValue();
        if (auto *subExpr = value.getDefiningOp()) {
            if (failed(exportExpression(subExpr, ss))) {
                return mlir::failure();
            }
        } else {
            if (failed(exportParameterReference(value, ss))) {
                return mlir::failure();
            }
        }
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportScope(P4HIR::ScopeOp &scopeOp, ExtendedFormattedOStream &ss) {
        ss << scopeOp;
        ss.semicolon();
        for (auto &nestedOp : scopeOp.getOps()) {
            if (auto extractStructRefOp = mlir::dyn_cast<P4HIR::StructExtractRefOp>(nestedOp)) {
                // TODO: Skip these for now. What to do with them?
                continue;
            }
            if (auto callOp = mlir::dyn_cast<P4HIR::CallMethodOp>(nestedOp)) {
                auto callee = callOp.getCallee();
                ss << callee.getRootReference().strref();
                for (auto calleeRef : callee.getNestedReferences()) {
                    ss << ".";
                    ss << calleeRef.getValue();
                }
                ss << "(";
                auto result = Utilities::interleaveCommaWithError(
                    callOp.getArgOperands(), ss,
                    [&](mlir::Value arg) { return exportExpression(arg.getDefiningOp(), ss); });
                if (failed(result)) {
                    return mlir::failure();
                }

                ss << ")";
                continue;
            }
            if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(nestedOp)) {
                // TODO: Skip these for now. What to do with them?
                continue;
            }
            if (auto variableOp = mlir::dyn_cast<P4HIR::VariableOp>(nestedOp)) {
                // TODO: Skip these for now. What to do with them?
                continue;
            }
            if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(nestedOp)) {
                // TODO: Skip these for now. What to do with them?
                continue;
            }
            if (auto yieldOp = mlir::dyn_cast<P4HIR::YieldOp>(nestedOp)) {
                // TODO: Skip these for now. What to do with them?
                continue;
            }
            return nestedOp.emitError() << "Unhandled scope statement: " << nestedOp;
        }
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportParserStateStatement(mlir::Operation &op,
                                                   ExtendedFormattedOStream &ss) {
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(exportAssignmentStatement(assignOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(exportP4Declaration(*varOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            // if (failed(exportConstant(constOp, ss))) {
            //     return mlir::failure();
            // }
            // ss.semicolon();
            return mlir::success();
        }
        if (auto castOP = mlir::dyn_cast<P4HIR::CastOp>(op)) {
            return mlir::success();
        }
        if (auto transitionOp = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            ss << "transition " << transitionOp.getState().getLeafReference().strref();
            ss.semicolon();
            return mlir::success();
        }
        if (auto parserAcceptOp = mlir::dyn_cast<P4HIR::ParserAcceptOp>(op)) {
            return mlir::success();
        }
        if (auto parserRejectOp = mlir::dyn_cast<P4HIR::ParserRejectOp>(op)) {
            return mlir::success();
        }
        if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(op)) {
            if (failed(exportScope(scopeOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }

        return op.emitError() << "Unhandled parser state statement: " << op;
    }

    mlir::LogicalResult exportParserStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        /// Assignments are not supported at the parser top-level. So we have to emit the type here
        /// and treat this as declaration. At the same time, we skip the variable declaration.
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            return mlir::success();
        }
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(exportAssigningDeclaration(assignOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto parserState = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
            auto parserStateName = parserState.getName();
            if (parserStateName == "accept" || parserStateName == "reject") {
                return mlir::success();
            }
            ss << "state " << parserState.getName() << " ";
            ss.openBrace();
            for (auto &nestedOp : parserState.getOps()) {
                if (failed(exportParserStateStatement(nestedOp, ss))) {
                    return mlir::failure();
                }
            }
            ss.closeBrace();
            return mlir::success();
        }
        if (auto parserTransition = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            return mlir::success();
        }
        return op.emitError() << "Unhandled parser statement: " << op;
    }

    mlir::LogicalResult exportParserDeclaration(P4HIR::ParserOp &parserOp,
                                                ExtendedFormattedOStream &ss) {
        ss << "parser " << parserOp.getName() << "(";
        if (failed(exportParameterList(parserOp, ss))) {
            return mlir::failure();
        }
        ss << ") ";
        ss.openBrace();
        for (auto &nestedOp : parserOp.getOps()) {
            if (failed(exportParserStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult exportControlApplyStatement(mlir::Operation &op,
                                                    ExtendedFormattedOStream &ss) {
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(exportAssignmentStatement(assignOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(exportP4Declaration(*varOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        return op.emitWarning() << " Control apply statement not implemented: " << op,
               mlir::success();
    }

    mlir::LogicalResult exportControlApply(P4HIR::ControlApplyOp &op,
                                           ExtendedFormattedOStream &ss) {
        ss << "apply ";
        ss.openBrace();
        for (auto &nestedOp : op.getOps()) {
            if (failed(exportControlApplyStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult exportControlStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(exportLValue(*assignOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss << " = ";
            auto value = assignOp.getValue();
            if (auto *subExpr = value.getDefiningOp()) {
                if (failed(exportExpression(subExpr, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(exportParameterReference(value, ss))) {
                    return mlir::failure();
                }
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(exportP4Declaration(*varOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto controlApplyOp = mlir::dyn_cast<P4HIR::ControlApplyOp>(op)) {
            if (failed(exportControlApply(controlApplyOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        return op.emitWarning() << " Control statement not implemented: " << op, mlir::success();
    }

    mlir::LogicalResult exportControlDeclaration(P4HIR::ControlOp &controlOp,
                                                 ExtendedFormattedOStream &ss) {
        ss << "control " << controlOp.getName() << "(";
        if (failed(exportParameterList(controlOp, ss))) {
            return mlir::failure();
        }
        ss << ") ";
        ss.openBrace();
        for (auto &nestedOp : controlOp.getOps()) {
            if (failed(exportControlStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    static mlir::LogicalResult exportParameterReference(mlir::Value value,
                                                        ExtendedFormattedOStream &ss) {
        if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
            unsigned argIndex = blockArg.getArgNumber();
            mlir::Block *ownerBlock = blockArg.getOwner();
            mlir::Operation *parentOp = ownerBlock->getParentOp();

            if (auto fun = mlir::dyn_cast<mlir::FunctionOpInterface>(parentOp)) {
                auto nameAttr = fun.getArgAttr(argIndex, P4HIR::FuncOp::getParamNameAttrName());
                if (nameAttr) {
                    if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                        ss << nameStrAttr.getValue();
                    } else {
                        return fun->emitError()
                               << "Param name attribute has unexpected type: " << nameAttr;
                    }
                } else {
                    return fun->emitError()
                           << "Missing param name attribute for function arg " << argIndex;
                }
            }

            return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(value.getContext()))
               << "Unhandled value: " << value;
    }

    mlir::LogicalResult getOriginalParamName(P4HIR::FuncOp funcOp, unsigned index,
                                             ExtendedFormattedOStream &ss) {
        auto argAttrs = funcOp.getArgAttrsAttr();
        auto typeAttrs = funcOp.getArgumentTypes();
        auto paramNameAttrName = P4HIR::FuncOp::getParamNameAttrName();

        if (!argAttrs && index >= argAttrs.size()) {
            return funcOp.emitError() << " Invalid arg index: " << index;
        }
        auto dict = argAttrs[index].dyn_cast_or_null<mlir::DictionaryAttr>();
        if (!dict) {
            return funcOp.emitError() << "Attribute is not a dictionary: " << index;
        }
        auto nameAttr = dict.get(paramNameAttrName).dyn_cast_or_null<mlir::StringAttr>();
        if (!nameAttr || nameAttr.getValue().empty()) {
            return funcOp.emitError() << "No name attribute for function arg " << index;
        }
        if (failed(exportP4Type(typeAttrs[index], ss))) {
            return mlir::failure();
        }
        ss << " " << nameAttr.getValue();
        return mlir::success();
    }

    mlir::LogicalResult exportTopLevelFuncOp(P4HIR::FuncOp &funcOp, ExtendedFormattedOStream &ss) {
        // Check whether this function was declared as part of an overload set.
        if (auto overloadSetOp = funcOp->getParentOfType<P4HIR::OverloadSetOp>()) {
            ss << "extern " << overloadSetOp.getName() << "(";
        } else {
            ss << "extern " << funcOp.getName() << "(";
        }
        unsigned numArgs = funcOp.getNumArguments();
        for (unsigned i = 0; i < numArgs; ++i) {
            if (i > 0) {
                ss << ", ";
            }

            P4HIR::ParamDirection direction = funcOp.getArgumentDirection(i);
            switch (direction) {
                case P4HIR::ParamDirection::None:
                    break;
                default:
                    ss << P4HIR::stringifyEnum(direction) << " ";
            }
            if (failed(getOriginalParamName(funcOp, i, ss))) {
                return mlir::failure();
            }
        }
        ss << ")";
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportInstantiateOP(P4HIR::InstantiateOp instantiateOp,
                                            ExtendedFormattedOStream &ss) {
        auto definitionOpt = findInstantiatedOpDefinition(instantiateOp);
        if (failed(definitionOpt)) {
            return instantiateOp.emitError() << "Could not resolve definition";
        }
        auto *definition = *definitionOpt;
        // Use dyn_cast to safely check the type and work with the specific Op class
        if (auto packageDef = llvm::dyn_cast<P4HIR::PackageOp>(definition)) {
            if (failed(exportPackageOp(packageDef, ss))) {
                return mlir::failure();
            }

            auto resultType = instantiateOp.getResult().getType();
            auto packageType = resultType.dyn_cast<P4HIR::PackageType>();
            if (!packageType) {
                return instantiateOp.emitError() << "Could not resolve package type";
            }
            ss << packageType.getName() << "(";

            llvm::interleaveComma(instantiateOp.getArgOperands(), ss, [&](const auto &arg) {
                if (auto subInstantiateOp =
                        mlir::dyn_cast<P4HIR::InstantiateOp>(arg.getDefiningOp())) {
                    ss << subInstantiateOp.getName() << "()";
                }
            });
            ss << ") ";
            ss << instantiateOp.getName();
            ss.semicolon();
            return mlir::success();
        }
        if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(definition)) {
            if (options().mainPackageOnly) {
                if (failed(exportParserDeclaration(parserOp, ss))) {
                    return mlir::failure();
                }
            }
            return mlir::success();
        }
        if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(definition)) {
            if (options().mainPackageOnly) {
                if (failed(exportControlDeclaration(controlOp, ss))) {
                    return mlir::failure();
                }
            }
            return mlir::success();
        }
        if (auto externDef = llvm::dyn_cast<P4HIR::ExternOp>(definition)) {
            // TODO: Any reason we might have to handle this instantiation?
            return mlir::success();
        }
        return instantiateOp.emitError() << "Unexpected definition type resolved";
    }

    mlir::LogicalResult exportExternStatementOp(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op)) {
            if (failed(exportTopLevelFuncOp(funcOp, ss))) {
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

        return op.emitError() << "Extern operation not implemented: " << op, mlir::success();
    }

    mlir::LogicalResult exportExternOp(P4HIR::ExternOp &externOp, ExtendedFormattedOStream &ss) {
        ss << "extern " << externOp.getName() << " ";
        ss.openBrace();
        for (auto &nestedOp : externOp.getOps()) {
            if (failed(exportExternStatementOp(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

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

    mlir::LogicalResult exportProgrammableBlock(mlir::Type &type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::ParserType>([&](P4HIR::ParserType parserType) {
                ss << P4HIR::ParserType::getMnemonic() << " ";
                ss << parserType.getName();
                if (failed(exportTypeArguments(parserType.getTypeArguments(), ss))) {
                    return mlir::failure();
                }

                ss << "(";
                auto result = Utilities::interleaveCommaWithError(
                    parserType.getInputs(), ss,
                    [&](mlir::Type type) { return exportP4Type(type, ss); });
                if (failed(result)) {
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
                auto result = Utilities::interleaveCommaWithError(
                    controlType.getInputs(), ss,
                    [&](mlir::Type type) { return exportP4Type(type, ss); });
                if (failed(result)) {
                    return mlir::failure();
                }
                ss << ")";
                ss.semicolon();
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                mlir::Location loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Unsupported P4HIR type for P4 export: " << t;
            });
    }

    mlir::LogicalResult exportPackageOp(P4HIR::PackageOp &packageOp, ExtendedFormattedOStream &ss) {
        llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> ctorParamTypes;
        if (auto ctorTypeAttr = packageOp.getCtorTypeAttr()) {
            if (auto ctorType = ctorTypeAttr.getValue().dyn_cast<P4HIR::CtorType>()) {
                ctorParamTypes = ctorType.getInputs();
            } else {
                return packageOp.emitError()
                       << "packageOp has CtorTypeAttr but not a valid p4hir.CtorType";
            }
        }

        for (const auto &ctorParamType : ctorParamTypes) {
            mlir::Type paramType = ctorParamType.second;
            if (llvm::failed(exportProgrammableBlock(paramType, ss))) {
                return mlir::failure();
            }
        }

        ss << "package " << packageOp.getName();
        mlir::SmallVector<mlir::Type> ctrParamTypes;
        if (mlir::ArrayAttr typeParamsAttr = packageOp.getTypeParametersAttr()) {
            for (auto typeAttr : typeParamsAttr) {
                if (auto ta = typeAttr.dyn_cast<mlir::TypeAttr>()) {
                    ctrParamTypes.push_back(ta.getValue());
                }
            }
        }
        if (failed(exportTypeArguments(ctrParamTypes, ss))) {
            return mlir::failure();
        }
        ss << "(";

        mlir::ArrayAttr ctorArgAttrs = packageOp.getArgAttrsAttr();
        for (unsigned idx = 0; idx < ctorParamTypes.size(); ++idx) {
            if (idx > 0) {
                ss << ", ";
            }
            mlir::Type paramType = ctorParamTypes[idx].second;

            // Get the P4 string for the parameter type instance (e.g., "MyParser<H, M>")
            if (failed(exportP4Type(paramType, ss))) {
                return mlir::failure();
            }
            if (auto dictAttr = ctorArgAttrs[idx].dyn_cast_or_null<mlir::DictionaryAttr>()) {
                if (auto nameAttr = dictAttr.get(P4HIR::FuncOp::getParamNameAttrName())
                                        .dyn_cast_or_null<mlir::StringAttr>()) {
                    if (!nameAttr.getValue().empty()) {
                        ss << " " << nameAttr.getValue();
                    }
                }
            } else {
                return packageOp.emitError() << "CtorArgAttrs is not a dictionary";
            }
        }
        ss << ")";
        ss.semicolon();
        return mlir::success();
    }

    mlir::LogicalResult exportTopLevelStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
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
            if (failed(exportInstantiateOP(instantiateOp, ss))) {
                return mlir::failure();
            }
            ss.newline();
            return mlir::success();
        }
        if (auto packageOp = mlir::dyn_cast<P4HIR::PackageOp>(op)) {
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
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            ss.newline();
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(exportP4Declaration(*varOp.getRef().getDefiningOp(), ss))) {
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
        return op.emitError() << "Operation not implemented: " << op, mlir::success();
    }
};

}  // namespace

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
            return mlir::emitError(mlir::UnknownLoc::get(module.getContext()))
                   << "Failed to open file for writing: " << p4OutputFile.string();
        }

        if (failed(P4::P4MLIR::Utilities::exportP4HirToP4(module, outfile, {}))) {
            return mlir::failure();
        }
        if (outfile.has_error()) {
            outfile.close();
            return emitError(mlir::UnknownLoc::get(module.getContext()))
                   << "Failed to write to file: " + p4OutputFile.string();
        }
        outfile.close();
    } catch (const std::filesystem::filesystem_error &ex) {
        return mlir::emitError(mlir::UnknownLoc::get(module.getContext()))
               << "Filesystem error: " << ex.what();
    } catch (const std::exception &ex) {
        return mlir::emitError(mlir::UnknownLoc::get(module.getContext()))
               << "Standard exception: " << ex.what();
    } catch (...) {
        return mlir::emitError(mlir::UnknownLoc::get(module.getContext()))
               << "Unknown exception occurred.";
    }
    return mlir::success();
}
}  // namespace P4::P4MLIR::Utilities
