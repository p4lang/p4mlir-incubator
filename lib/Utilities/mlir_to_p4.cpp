#include "mlir_to_p4.h"

#include <string>

#include "formatted_stream.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#pragma GCC diagnostic pop
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR::Utilities {

namespace {

/// TODO: Keep this class and its API private until its API is finalized.
class MlirToP4Program {
 public:
    /// Converts the given MLIR module to a formatted P4 program string.
    std::string convert(mlir::ModuleOp module);

    /// Writes the given MLIR module to the given stream as a formatted P4 program.
    void convert(mlir::ModuleOp module, std::ostream &stream);

    void writeProgram(mlir::ModuleOp module, FormattedStream &ss) {
        module.walk([&](mlir::Operation *op) {
            if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
                llvm::outs() << "Converting module: " << subModule->getName() << "\n";
                for (auto &op : subModule.getOps()) {
                    llvm::outs() << "Converting operation: " << op.getName() << "\n";
                    // if (op.hasTrait<mlir::OpTrait::ZeroResults>()) {
                    convertStatement(op, ss);
                    // }
                }
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
    }

    explicit MlirToP4Program(MlirToP4ProgramOptions options) : options_(options) {}

 private:
    MlirToP4ProgramOptions options_;

    void complexTypeToP4String(const mlir::Type &type, FormattedStream &ss) {
        if (auto ref = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
            complexTypeToP4String(ref.getObjectType(), ss);
        } else if (auto headerType = llvm::dyn_cast<P4HIR::HeaderType>(type)) {
            ss << headerType.getName().str();
        } else if (auto structType = llvm::dyn_cast<P4HIR::StructType>(type)) {
            ss << structType.getName().str();
        } else if (auto headerUnion = llvm::dyn_cast<P4HIR::HeaderUnionType>(type)) {
            ss << headerUnion.getName().str();
        } else if (auto externType = llvm::dyn_cast<P4HIR::ExternType>(type)) {
            ss << externType.getName().str();
        } else {
            llvm::errs() << "Unsupported " << type << "\n";
            llvm::report_fatal_error("Encountered unhandled type.");
        }
    }

    void referenceToP4String(const mlir::Operation &type, FormattedStream &ss) {
        if (auto var = llvm::dyn_cast<P4HIR::StructExtractRefOp>(type)) {
            if (auto ref = var.getInput().getDefiningOp()) {
                referenceToP4String(*ref, ss);
            } else {
                complexTypeToP4String(var.getInput().getType(), ss);
            }
            ss << "." << var.getFieldName().str();
        } else {
            llvm::errs() << "Unsupported " << type << "\n";
        }
    }

    void constantToP4String(const P4HIR::ConstOp &constant, FormattedStream &ss) {
        for (auto namedAttr : constant->getAttrs()) {
            mlir::Attribute attrValue = namedAttr.getValue();
            if (auto intAttr = attrValue.dyn_cast<P4HIR::IntAttr>()) {
                auto bitType = llvm::dyn_cast<P4HIR::BitsType>(intAttr.getType());
                if (!bitType) {
                    llvm::errs() << "Expected integers to be typed at this point";
                    return;
                }
                auto width = bitType.getWidth();
                ss << width << "w";
                // TODO: Cleanup. How to handle this safely?
                llvm::SmallVector<char, 64> valueBuffer;
                intAttr.getValue().toString(valueBuffer, 10, false);
                std::string valueStr(valueBuffer.begin(), valueBuffer.end());
                ss << valueStr;
                return;
            } else if (auto strAttr = attrValue.dyn_cast<mlir::StringAttr>()) {
                ss << strAttr.getValue().str();
                return;
            } else {
                llvm::outs() << "  -> Unhandled Attribute Type: " << attrValue << "\n";
            }
        }
        // Consider adding error handling if no valid attribute is found
    }

    void convertExpression(mlir::Operation &op, FormattedStream &ss) {
        if (auto constOp = llvm::dyn_cast<P4HIR::ConstOp>(op)) {
            constantToP4String(constOp, ss);
        } else {
            llvm::errs() << "Unhandled operation: " << op.getName() << "\n";
        }
    }

    void convertControlApply(P4HIR::ControlApplyOp &op, FormattedStream &ss) {
        ss << "apply ";
        ss.openBrace();
        for (auto &nestedOp : op.getOps()) {
            convertStatement(nestedOp, ss);
        }
        ss.closeBrace();
    }

    void convertDirection(const P4HIR::ParamDirectionAttr &direction, FormattedStream &ss) {
        if (direction.getValue() == P4HIR::ParamDirection::In) {
            ss << "in ";
        } else if (direction.getValue() == P4HIR::ParamDirection::Out) {
            ss << "out ";
        } else if (direction.getValue() == P4HIR::ParamDirection::InOut) {
            ss << "inout ";
        } else if (direction.getValue() == P4HIR::ParamDirection::None) {
            // No direction keyword in the case of none.
        } else {
            llvm::errs() << "Unhandled direction: " << direction << "\n";
        }
    }

    void convertArgumentList(mlir::FunctionOpInterface functionInterface, FormattedStream &ss) {
        bool needsComma = false;

        for (auto arg : functionInterface.getArguments()) {
            if (needsComma) {
                ss << ", ";
            }
            auto direction = functionInterface.getArgAttr(arg.getArgNumber(),
                                                          P4HIR::FuncOp::getDirectionAttrName());
            auto castDirection = llvm::dyn_cast<P4HIR::ParamDirectionAttr>(direction);
            convertDirection(castDirection, ss);

            complexTypeToP4String(arg.getType(), ss);
            ss << " ";

            auto nameAttr = functionInterface.getArgAttr(arg.getArgNumber(),
                                                         P4HIR::FuncOp::getParamNameAttrName());
            if (nameAttr) {
                if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                    ss << nameStrAttr.getValue().str();
                } else {
                    llvm::errs() << "Param name attribute has unexpected type: " << nameAttr
                                 << "\n";
                }
            } else {
                llvm::errs() << "Missing param name attribute for function arg "
                             << arg.getArgNumber() << "\n";
                ss << "_param" << arg.getArgNumber();
            }
            needsComma = true;
        }
    }

    void convertParser(P4HIR::ParserOp &parserOp, FormattedStream &ss) {
        ss << "parser " << parserOp.getName().str() << "(";
        convertArgumentList(parserOp, ss);
        ss << ")";
        ss.openBrace();
        for (auto &nestedOp : parserOp.getOps()) {
            convertStatement(nestedOp, ss);
        }
        ss.closeBrace();
        ss.newline();
    }

    void convertControl(P4HIR::ControlOp &controlOp, FormattedStream &ss) {
        ss << "control " << controlOp.getName().str() << "(";
        convertArgumentList(controlOp, ss);
        ss << ")";
        ss.openBrace();
        for (auto &nestedOp : controlOp.getOps()) {
            convertStatement(nestedOp, ss);
        }
        ss.closeBrace();
        ss.newline();
    }

    void convertStatement(mlir::Operation &op, FormattedStream &ss) {
        if (auto funcOp = llvm::dyn_cast<P4HIR::FuncOp>(op)) {
            // TODO: Implement FuncOp conversion.
        } else if (auto packageOp = llvm::dyn_cast<P4HIR::PackageOp>(op)) {
            // TODO: Implement PackageOp conversion.
        } else if (auto assignOp = llvm::dyn_cast<P4HIR::AssignOp>(op)) {
            referenceToP4String(*assignOp.getRef().getDefiningOp(), ss);
            ss << " = ";
            convertExpression(*assignOp.getValue().getDefiningOp(), ss);
            ss.semicolon();
        } else if (auto controlApplyOp = llvm::dyn_cast<P4HIR::ControlApplyOp>(op)) {
            convertControlApply(controlApplyOp, ss);
        } else if (auto parserOp = llvm::dyn_cast<P4HIR::ParserOp>(op)) {
            convertParser(parserOp, ss);
        } else if (auto controlOp = llvm::dyn_cast<P4HIR::ControlOp>(op)) {
            convertControl(controlOp, ss);
        } else if (auto instantiateOp = llvm::dyn_cast<P4HIR::InstantiateOp>(op)) {
            // TODO: Implement InstantiateOp conversion.
        } else if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
            llvm::errs() << "Not sure what to do with nested module op yet" << "\n";
        } else {
            llvm::errs() << "Unhandled operation: " << op << "\n";
            // llvm::report_fatal_error("Encountered unhandled operation.");
        }
    }
};

}  // namespace

std::string convertMlirToP4(mlir::ModuleOp module, MlirToP4ProgramOptions options) {
    MlirToP4Program converter(options);
    FormattedStream ss(options.indent_unit);
    converter.writeProgram(module, ss);
    return ss.str();
}

void convertMlirToP4(mlir::ModuleOp module, std::ostream &os, MlirToP4ProgramOptions options) {
    MlirToP4Program converter(options);
    FormattedStream ss(os, options.indent_unit);
    converter.writeProgram(module, ss);
}

}  // namespace P4::P4MLIR::Utilities
