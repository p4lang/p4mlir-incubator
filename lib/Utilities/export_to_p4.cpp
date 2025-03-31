#include "export_to_p4.h"

#include <string>
#include <system_error>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
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

        // // Collect types mentioned in operation attributes
        // for (mlir::NamedAttribute namedAttr : op->getAttrs()) {
        //     findTypesInAttribute(namedAttr.getValue(), collectedTypes);
        // }
    });

    return collectedTypes;
}

/// TODO: Keep this class private until its API is finalized.
class P4HirToP4Exporter {
 public:
    /// Converts the given MLIR module to a formatted P4 program string.
    llvm::Expected<std::string> convert(mlir::ModuleOp module);

    /// Writes the given MLIR module to the given stream as a formatted P4 program.
    mlir::LogicalResult convert(mlir::ModuleOp module, std::ostream &stream);

    mlir::LogicalResult writeProgram(mlir::ModuleOp module, ExtendedFormattedOStream &ss) {
        // 1. Try to find all the used types, and declare all complex types.
        llvm::SetVector<mlir::Type> allTypes = getAllUsedTypes(module);
        for (mlir::Type type : allTypes) {
            // llvm::outs() << "Type: " << type << "\n";
            if (auto structType = llvm::dyn_cast<P4HIR::StructLikeTypeInterface>(type)) {
                // llvm::outs() << "Struct: " << structType << "\n";
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
            } else if (auto refType = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
                auto subType = refType.getObjectType();
                if (subType.isa<P4HIR::StructLikeTypeInterface>()) {
                    if (failed(declareType(subType, ss))) {
                        return mlir::failure();
                    }
                }
            }
        }

        // 2. Generate the programmable blocks and the package.
        for (auto &op : module.getOps()) {
            if (op.hasTrait<mlir::OpTrait::ZeroResults>()) {
                if (failed(convertTopLevelStatement(op, ss))) {
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    explicit P4HirToP4Exporter(P4HirToP4ExporterOptions options) : options_(options) {}

 private:
    P4HirToP4ExporterOptions options_;

    mlir::LogicalResult typeToP4String(mlir::Type type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::ReferenceType>([&](P4HIR::ReferenceType ref) {
                // Recursively get the underlying object type's name
                return typeToP4String(ref.getObjectType(), ss);
            })
            .Case<P4HIR::HeaderType>([&](P4HIR::HeaderType headerType) {
                ss << headerType.getName().str();
                return mlir::success();
            })
            .Case<P4HIR::StructType>([&](P4HIR::StructType structType) {
                ss << structType.getName().str();
                return mlir::success();
            })
            .Case<P4HIR::HeaderUnionType>([&](P4HIR::HeaderUnionType unionType) {
                ss << unionType.getName().str();
                return mlir::success();
            })
            .Case<P4HIR::ExternType>([&](P4HIR::ExternType externType) {
                ss << externType.getName().str();
                return mlir::success();
            })
            .Case<P4HIR::BitsType>([&](P4HIR::BitsType bits) {
                ss << "bit<" << bits.getWidth() << ">";
                return mlir::success();
            })
            .Case<P4HIR::BoolType>([&](P4HIR::BoolType /*boolTy*/) {
                ss << "bool";
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                mlir::Location loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Unsupported P4HIR type for P4 emission: " << t;
            });
    }

    mlir::LogicalResult declareType(mlir::Type type, ExtendedFormattedOStream &ss) {
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<P4HIR::HeaderType>([&](P4HIR::HeaderType headerType) {
                ss << headerType.getMnemonic().data() << " " << headerType.getName().str() << " ";
                ss.openBrace();
                for (auto field : headerType.getFields()) {
                    if (failed(typeToP4String(field.type, ss))) {
                        return mlir::failure();
                    }
                    ss << " " << field.name.data() << ";";
                    ss.newline();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Case<P4HIR::StructType>([&](P4HIR::StructType structType) {
                ss << structType.getMnemonic().data() << " " << structType.getName().str() << " ";
                ss.openBrace();
                for (auto field : structType.getFields()) {
                    if (failed(typeToP4String(field.type, ss))) {
                        return mlir::failure();
                    }
                    ss << " " << field.name.data() << ";";
                    ss.newline();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Case<P4HIR::HeaderUnionType>([&](P4HIR::HeaderUnionType unionType) {
                ss << unionType.getMnemonic().data() << " " << unionType.getName().str() << " ";
                ss.openBrace();
                for (auto field : unionType.getFields()) {
                    if (failed(typeToP4String(field.type, ss))) {
                        return mlir::failure();
                    }
                    ss << " " << field.name.data() << ";";
                    ss.newline();
                }
                ss.closeBrace();
                return mlir::success();
            })
            .Default([&](mlir::Type t) {
                mlir::Location loc = mlir::UnknownLoc::get(t.getContext());
                return mlir::emitError(loc) << "Type declaration unsupported for: " << t;
            });
    }

    mlir::LogicalResult lValueToP4String(mlir::Operation &type, ExtendedFormattedOStream &ss) {
        if (auto var = mlir::dyn_cast<P4HIR::StructExtractRefOp>(type)) {
            if (auto ref = var.getInput().getDefiningOp()) {
                if (failed(lValueToP4String(*ref, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(convertParameterReference(var.getInput(), ss))) {
                    return mlir::failure();
                }
            }
            ss << "." << var.getFieldName().str();
            return mlir::success();
        } else if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(type)) {
            if (auto varName = var.getName()) {
                ss << varName.value().str();
            } else {
                return var.emitError() << "Variable declaration without name?";
            }
            return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(type.getContext()))
               << "Unsupported lvalue " << type;
    }

    mlir::LogicalResult declarationToP4String(mlir::Operation &type, ExtendedFormattedOStream &ss) {
        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(type)) {
            if (failed(typeToP4String(var.getType(), ss))) {
                return mlir::failure();
            }
            if (failed(lValueToP4String(type, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(type.getContext()))
               << "Unsupported declaration " << type;
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

    mlir::LogicalResult constantToP4String(P4HIR::ConstOp &constant, ExtendedFormattedOStream &ss) {
        mlir::Attribute attrValue = constant.getValueAttr();
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

        } else if (auto strAttr = attrValue.dyn_cast<mlir::StringAttr>()) {
            ss << escapeP4String(strAttr.getValue());
            return mlir::success();

        } else if (auto boolAttr = attrValue.dyn_cast<mlir::BoolAttr>()) {
            ss << (boolAttr.getValue() ? "true" : "false");
            return mlir::success();
        }

        return mlir::emitError(loc) << "Unsupported attribute type for P4 constant: " << attrValue;
    }

    mlir::LogicalResult convertExpression(mlir::Operation *op, ExtendedFormattedOStream &ss) {
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            return constantToP4String(constOp, ss);
        }
        if (auto cast = mlir::dyn_cast<P4HIR::CastOp>(op)) {
            ss << "(";
            if (failed(typeToP4String(cast.getType(), ss))) {
                return mlir::failure();
            }
            ss << ") ";
            if (failed(convertExpression(cast.getSrc().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        return op->emitError() << "Unhandled expression: " << *op;
    }

    mlir::LogicalResult convertControlApply(P4HIR::ControlApplyOp &op,
                                            ExtendedFormattedOStream &ss) {
        ss << "apply ";
        ss.openBrace();
        for (auto &nestedOp : op.getOps()) {
            if (failed(convertTopLevelStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult convertDirection(const P4HIR::ParamDirectionAttr &direction,
                                         ExtendedFormattedOStream &ss) {
        switch (direction.getValue()) {
            case P4HIR::ParamDirection::None:
                return mlir::success();
            default:
                ss << P4HIR::stringifyEnum(direction.getValue()).data() << " ";
                return mlir::success();
        }
        return mlir::emitError(mlir::UnknownLoc::get(direction.getContext()))
               << "Unhandled direction: " << direction;
    }

    mlir::LogicalResult convertArgument(mlir::FunctionOpInterface functionInterface,
                                        ExtendedFormattedOStream &ss, int index) {
        auto argument = functionInterface.getArgument(index);
        auto direction = functionInterface.getArgAttr(index, P4HIR::FuncOp::getDirectionAttrName());
        auto castDirection = mlir::dyn_cast<P4HIR::ParamDirectionAttr>(direction);
        if (failed(convertDirection(castDirection, ss))) {
            return mlir::failure();
        }
        if (failed(typeToP4String(argument.getType(), ss))) {
            return mlir::failure();
        }

        ss << " ";

        auto nameAttr = functionInterface.getArgAttr(index, P4HIR::FuncOp::getParamNameAttrName());
        if (nameAttr) {
            if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                ss << nameStrAttr.getValue().str();
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

    mlir::LogicalResult convertArgumentList(mlir::FunctionOpInterface functionInterface,
                                            ExtendedFormattedOStream &ss) {
        bool needsComma = false;

        for (auto arg : functionInterface.getArguments()) {
            if (needsComma) {
                ss << ", ";
            }
            if (failed(convertArgument(functionInterface, ss, arg.getArgNumber()))) {
                return mlir::failure();
            }
            needsComma = true;
        }
        return mlir::success();
    }

    mlir::LogicalResult convertParserStateStatement(mlir::Operation &op,
                                                    ExtendedFormattedOStream &ss) {
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(lValueToP4String(*assignOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss << " = ";
            auto value = assignOp.getValue();
            if (auto subExpr = value.getDefiningOp()) {
                if (failed(convertExpression(subExpr, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(convertParameterReference(value, ss))) {
                    return mlir::failure();
                }
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(declarationToP4String(*varOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op)) {
            // if (failed(constantToP4String(constOp, ss))) {
            //     return mlir::failure();
            // }
            // ss.semicolon();
            return mlir::success();
        }
        if (auto castOP = mlir::dyn_cast<P4HIR::CastOp>(op)) {
            return mlir::success();
        }
        if (auto transitionOp = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            ss << "transition " << transitionOp.getState().getLeafReference().data();
            ss.semicolon();
            return mlir::success();
        }
        if (auto parserAcceptOp = mlir::dyn_cast<P4HIR::ParserAcceptOp>(op)) {
            return mlir::success();
        }
        if (auto parserRejectOp = mlir::dyn_cast<P4HIR::ParserRejectOp>(op)) {
            return mlir::success();
        }
        return op.emitError() << "Unhandled parser state statement: " << op;
    }

    mlir::LogicalResult convertParserStatement(mlir::Operation &op, ExtendedFormattedOStream &ss) {
        /// Assignments are not supported at the parser top-level. So we have to emit the type here
        /// and treat this as declaration. At the same time, we skip the variable declaration.
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            return mlir::success();
        }
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(declarationToP4String(*assignOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss << " = ";
            auto value = assignOp.getValue();
            if (auto subExpr = value.getDefiningOp()) {
                if (failed(convertExpression(subExpr, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(convertParameterReference(value, ss))) {
                    return mlir::failure();
                }
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto parserState = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
            auto parserStateName = parserState.getName();
            if (parserStateName == "accept" || parserStateName == "reject") {
                return mlir::success();
            }
            ss << "state " << parserState.getName().str() << " ";
            ss.openBrace();
            for (auto &nestedOp : parserState.getOps()) {
                if (failed(convertParserStateStatement(nestedOp, ss))) {
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

    mlir::LogicalResult convertParser(P4HIR::ParserOp &parserOp, ExtendedFormattedOStream &ss) {
        ss << "parser " << parserOp.getName().str() << "(";
        if (failed(convertArgumentList(parserOp, ss))) {
            return mlir::failure();
        }
        ss << ") ";
        ss.openBrace();
        for (auto &nestedOp : parserOp.getOps()) {
            if (failed(convertParserStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult convertControl(P4HIR::ControlOp &controlOp, ExtendedFormattedOStream &ss) {
        ss << "control " << controlOp.getName().str() << "(";
        if (failed(convertArgumentList(controlOp, ss))) {
            return mlir::failure();
        }
        ss << ") ";
        ss.openBrace();
        for (auto &nestedOp : controlOp.getOps()) {
            if (failed(convertTopLevelStatement(nestedOp, ss))) {
                return mlir::failure();
            }
        }
        ss.closeBrace();
        return mlir::success();
    }

    mlir::LogicalResult convertParameterReference(mlir::Value value, ExtendedFormattedOStream &ss) {
        if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
            unsigned argIndex = blockArg.getArgNumber();
            mlir::Block *ownerBlock = blockArg.getOwner();
            mlir::Operation *parentOp = ownerBlock->getParentOp();

            if (auto fun = mlir::dyn_cast<mlir::FunctionOpInterface>(parentOp)) {
                auto nameAttr = fun.getArgAttr(argIndex, P4HIR::FuncOp::getParamNameAttrName());
                if (nameAttr) {
                    if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                        ss << nameStrAttr.getValue().str();
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

    mlir::LogicalResult convertTopLevelStatement(mlir::Operation &op,
                                                 ExtendedFormattedOStream &ss) {
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op)) {
            // TODO: Implement FuncOp conversion.
        }
        if (auto packageOp = mlir::dyn_cast<P4HIR::PackageOp>(op)) {
            // TODO: Implement PackageOp conversion.
        }
        if (auto externOp = mlir::dyn_cast<P4HIR::ExternOp>(op)) {
            // TODO: Implement ExternOp conversion.
        }
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            if (failed(lValueToP4String(*assignOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss << " = ";
            auto value = assignOp.getValue();
            if (auto subExpr = value.getDefiningOp()) {
                if (failed(convertExpression(subExpr, ss))) {
                    return mlir::failure();
                }
            } else {
                if (failed(convertParameterReference(value, ss))) {
                    return mlir::failure();
                }
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto varOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            if (failed(declarationToP4String(*varOp.getRef().getDefiningOp(), ss))) {
                return mlir::failure();
            }
            ss.semicolon();
            return mlir::success();
        }
        if (auto controlApplyOp = mlir::dyn_cast<P4HIR::ControlApplyOp>(op)) {
            if (failed(convertControlApply(controlApplyOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
            if (failed(convertParser(parserOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
            if (failed(convertControl(controlOp, ss))) {
                return mlir::failure();
            }
            return mlir::success();
        }
        if (auto instantiateOp = mlir::dyn_cast<P4HIR::InstantiateOp>(op)) {
            // TODO: Implement InstantiateOp conversion
        }
        if (auto transitionOp = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            // TODO: Implement ParserTransitionOp conversion?
            return mlir::success();
        }
        if (auto subModule = mlir::dyn_cast<mlir::ModuleOp>(op)) {
            return subModule->emitWarning() << "Not sure what to do with nested module op yet",
                   mlir::success();
        }
        return op.emitWarning() << "Operation not implemented: " << op, mlir::success();
    }
};

}  // namespace

llvm::FailureOr<std::string> exportP4HirToP4(mlir::ModuleOp module,
                                             P4HirToP4ExporterOptions options) {
    P4HirToP4Exporter converter(options);
    std::string outputString;
    llvm::raw_string_ostream rso(outputString);
    ExtendedFormattedOStream ss(rso, options.style, options.indentLevel);
    if (failed(converter.writeProgram(module, ss))) {
        return mlir::failure();
    }
    return rso.str();
}

mlir::LogicalResult exportP4HirToP4(mlir::ModuleOp module, llvm::raw_ostream &os,
                                    P4HirToP4ExporterOptions options) {
    P4HirToP4Exporter converter(options);
    ExtendedFormattedOStream ss(os, options.style, options.indentLevel);
    if (failed(converter.writeProgram(module, ss))) {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult writeP4HirToP4File(mlir::ModuleOp module, std::filesystem::path p4OutputFile) {
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
