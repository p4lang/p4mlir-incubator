#include "mlir_to_p4.h"

#include <sstream>

#include "ir/ir.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR {

MlirToP4::MlirToP4(mlir::ModuleOp module) : module(module) {}

namespace {

std::string complexTypeToP4String(const mlir::Type &type) {
    std::stringstream ss;
    if (auto ref = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
        ss << complexTypeToP4String(ref.getObjectType());
    } else if (auto headerType = llvm::dyn_cast<P4HIR::HeaderType>(type)) {
        ss << headerType.getName().str();
    } else if (auto structType = llvm::dyn_cast<P4HIR::StructType>(type)) {
        ss << structType.getName().str();
    } else if (auto headerUnion = llvm::dyn_cast<P4HIR::HeaderUnionType>(type)) {
        ss << headerUnion.getName().str();
    } else {
        llvm::errs() << "Unsupported " << type << "\n";
    }
    return ss.str();
}

std::string referenceToP4String(const mlir::Operation &type) {
    std::stringstream ss;
    if (auto var = llvm::dyn_cast<P4HIR::StructExtractRefOp>(type)) {
        if (auto ref = var.getInput().getDefiningOp()) {
            ss << referenceToP4String(*ref);
        } else {
            ss << complexTypeToP4String(var.getInput().getType());
        }
        ss << "." << var.getFieldName().str();
    } else {
        llvm::errs() << "Unsupported " << type << "\n";
    }

    return ss.str();
}

std::string constantToP4String(const P4HIR::ConstOp &constant) {
    std::stringstream ss;
    for (auto namedAttr : constant->getAttrs()) {
        mlir::Attribute attrValue = namedAttr.getValue();
        if (auto intAttr = attrValue.dyn_cast<P4HIR::IntAttr>()) {
            // TODO: Cleanup.
            llvm::SmallVector<char, 64> valueBuffer;
            intAttr.getValue().toString(valueBuffer, 10, false);
            std::string valueStr(valueBuffer.begin(), valueBuffer.end());
            ss << valueStr;
            return ss.str();
        } else if (auto strAttr = attrValue.dyn_cast<mlir::StringAttr>()) {
            ss << strAttr.getValue().str();
            return ss.str();
        } else {
            llvm::outs() << "  -> Unhandled Attribute Type: " << attrValue << "\n";
        }
    }

    return ss.str();
}

std::string convertExpression(mlir::Operation *op) {
    llvm::outs() << "#### " << op->getName() << " ####\n";
    // llvm::outs() << op->getLoc() << "\n";
    std::stringstream ss;

    if (auto funcOp = llvm::dyn_cast<P4HIR::FuncOp>(op)) {
    } else if (auto overloadSetOp = llvm::dyn_cast<P4HIR::OverloadSetOp>(op)) {
    } else if (auto externOp = llvm::dyn_cast<P4HIR::ExternOp>(op)) {
    } else if (auto constOp = llvm::dyn_cast<P4HIR::ConstOp>(op)) {
        ss << constantToP4String(constOp);
    } else if (auto packageOp = llvm::dyn_cast<P4HIR::PackageOp>(op)) {
    } else if (auto structExtractRefOp = llvm::dyn_cast<P4HIR::StructExtractRefOp>(op)) {
    } else if (auto variableOp = llvm::dyn_cast<P4HIR::VariableOp>(op)) {
    } else if (auto callMethodOp = llvm::dyn_cast<P4HIR::CallMethodOp>(op)) {
    } else if (auto readOp = llvm::dyn_cast<P4HIR::ReadOp>(op)) {
    } else if (auto yieldOp = llvm::dyn_cast<P4HIR::YieldOp>(op)) {
    } else if (auto scopeOp = llvm::dyn_cast<P4HIR::ScopeOp>(op)) {
    } else if (auto stateOp = llvm::dyn_cast<P4HIR::ParserStateOp>(op)) {
    } else if (auto transitionOp = llvm::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
    } else if (auto parserAcceptOp = llvm::dyn_cast<P4HIR::ParserAcceptOp>(op)) {
    } else if (auto parserRejectOp = llvm::dyn_cast<P4HIR::ParserRejectOp>(op)) {
    } else if (auto parserOp = llvm::dyn_cast<P4HIR::ParserOp>(op)) {
    } else if (auto controlApplyOp = llvm::dyn_cast<P4HIR::ControlApplyOp>(op)) {
    } else if (auto controlOp = llvm::dyn_cast<P4HIR::ControlOp>(op)) {
    } else if (auto structExtractOp = llvm::dyn_cast<P4HIR::StructExtractOp>(op)) {
    } else if (auto cmpOp = llvm::dyn_cast<P4HIR::CmpOp>(op)) {
    } else if (auto castOp = llvm::dyn_cast<P4HIR::CastOp>(op)) {
    } else if (auto implicitReturnOp = llvm::dyn_cast<P4HIR::ImplicitReturnOp>(op)) {
    } else if (auto callOp = llvm::dyn_cast<P4HIR::CallOp>(op)) {
    } else if (auto tableActionOp = llvm::dyn_cast<P4HIR::TableActionOp>(op)) {
    } else if (auto tableActionsOp = llvm::dyn_cast<P4HIR::TableActionsOp>(op)) {
    } else if (auto tableApplyOp = llvm::dyn_cast<P4HIR::TableApplyOp>(op)) {
    } else if (auto tableDefaultActionOp = llvm::dyn_cast<P4HIR::TableDefaultActionOp>(op)) {
    } else if (auto tableOp = llvm::dyn_cast<P4HIR::TableOp>(op)) {
    } else if (auto instantiateOp = llvm::dyn_cast<P4HIR::InstantiateOp>(op)) {
    } else if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        llvm::errs() << "Not sure what to do with module yet" << "\n";
    } else {
        llvm::errs() << "Unhandled operation: " << op->getName() << "\n";
        llvm::report_fatal_error("Encountered unhandled operation.");
    }
    // llvm::outs() << ss.str();
    return ss.str();
}

std::string convertOperation(mlir::Operation *op) {
    llvm::outs() << "#### " << op->getName() << " ####\n";
    // llvm::outs() << op->getLoc() << "\n";
    std::stringstream ss;

    if (auto funcOp = llvm::dyn_cast<P4HIR::FuncOp>(op)) {
    } else if (auto overloadSetOp = llvm::dyn_cast<P4HIR::OverloadSetOp>(op)) {
    } else if (auto externOp = llvm::dyn_cast<P4HIR::ExternOp>(op)) {
    } else if (auto constOp = llvm::dyn_cast<P4HIR::ConstOp>(op)) {
    } else if (auto packageOp = llvm::dyn_cast<P4HIR::PackageOp>(op)) {
    } else if (auto structExtractRefOp = llvm::dyn_cast<P4HIR::StructExtractRefOp>(op)) {
    } else if (auto variableOp = llvm::dyn_cast<P4HIR::VariableOp>(op)) {
    } else if (auto callMethodOp = llvm::dyn_cast<P4HIR::CallMethodOp>(op)) {
    } else if (auto readOp = llvm::dyn_cast<P4HIR::ReadOp>(op)) {
    } else if (auto assignOp = llvm::dyn_cast<P4HIR::AssignOp>(op)) {
        auto result = referenceToP4String(*assignOp.getRef().getDefiningOp());
        ss << referenceToP4String(*assignOp.getRef().getDefiningOp());
        ss << " = ";
        ss << convertExpression(assignOp.getValue().getDefiningOp());
        llvm::outs() << "RESUILT:222 " << ss.str() << "\n";
        ss << "\n";
    } else if (auto yieldOp = llvm::dyn_cast<P4HIR::YieldOp>(op)) {
    } else if (auto scopeOp = llvm::dyn_cast<P4HIR::ScopeOp>(op)) {
    } else if (auto stateOp = llvm::dyn_cast<P4HIR::ParserStateOp>(op)) {
    } else if (auto transitionOp = llvm::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
    } else if (auto parserAcceptOp = llvm::dyn_cast<P4HIR::ParserAcceptOp>(op)) {
    } else if (auto parserRejectOp = llvm::dyn_cast<P4HIR::ParserRejectOp>(op)) {
    } else if (auto parserOp = llvm::dyn_cast<P4HIR::ParserOp>(op)) {
    } else if (auto controlApplyOp = llvm::dyn_cast<P4HIR::ControlApplyOp>(op)) {
    } else if (auto controlOp = llvm::dyn_cast<P4HIR::ControlOp>(op)) {
    } else if (auto structExtractOp = llvm::dyn_cast<P4HIR::StructExtractOp>(op)) {
    } else if (auto cmpOp = llvm::dyn_cast<P4HIR::CmpOp>(op)) {
    } else if (auto castOp = llvm::dyn_cast<P4HIR::CastOp>(op)) {
    } else if (auto implicitReturnOp = llvm::dyn_cast<P4HIR::ImplicitReturnOp>(op)) {
    } else if (auto callOp = llvm::dyn_cast<P4HIR::CallOp>(op)) {
    } else if (auto tableActionOp = llvm::dyn_cast<P4HIR::TableActionOp>(op)) {
    } else if (auto tableActionsOp = llvm::dyn_cast<P4HIR::TableActionsOp>(op)) {
    } else if (auto tableApplyOp = llvm::dyn_cast<P4HIR::TableApplyOp>(op)) {
    } else if (auto tableDefaultActionOp = llvm::dyn_cast<P4HIR::TableDefaultActionOp>(op)) {
    } else if (auto tableOp = llvm::dyn_cast<P4HIR::TableOp>(op)) {
    } else if (auto instantiateOp = llvm::dyn_cast<P4HIR::InstantiateOp>(op)) {
    } else if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        llvm::errs() << "Not sure what to do with module yet" << "\n";
    } else {
        llvm::errs() << "Unhandled operation: " << op->getName() << "\n";
        llvm::report_fatal_error("Encountered unhandled operation.");
    }
    // llvm::outs() << ss.str();
    return ss.str();
}

}  // namespace

std::string MlirToP4::convert() {
    std::string p4Code;
    module.walk([&](mlir::Operation *op) {
        if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
            llvm::outs() << "Converting module: " << subModule->getName() << "\n";
            subModule.walk(
                [&](mlir::Operation *moduleOp) { p4Code += convertOperation(moduleOp); });
        }
    });
    return p4Code;
}

}  // namespace P4::P4MLIR
