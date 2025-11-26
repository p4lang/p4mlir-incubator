#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

#include "mlir/IR/Builders.h"
#include "p4mlir//Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_OpInterfaces.h"

using namespace mlir;
using namespace P4::P4MLIR::BMv2IR;

LogicalResult ParserStateOp::verify() {
    // Check that only allowed ops are used as transition keys, transitions and parser ops

    for (auto &block : getTransitionKeys()) {
        for (auto &op : block) {
            if (!isa<AllowedTransitionKey>(op)) {
                return emitError("Op not allowed as transition key");
            }
        }
    }

    for (auto &block : getTransitions()) {
        for (auto &op : block) {
            if (!isa<TransitionOp>(op)) return emitError("Op not allowed as transition");
        }
    }

    for (auto &block : getParserOps()) {
        for (auto &op : block) {
            if (!isa<AllowedParserOp>(op)) return emitError("Op not allowed as parser op");
        }
    }
    return success();
}

static mlir::ModuleOp getParentModule(Operation *from) {
    if (auto moduleOp = from->getParentOfType<mlir::ModuleOp>()) return moduleOp;

    from->emitOpError("could not find parent module op");
    return nullptr;
}

LogicalResult SymToValueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the decl attribute was specified.
    auto declAttr = (*this)->getAttrOfType<SymbolRefAttr>(getDeclAttrName());
    if (!declAttr) return emitOpError("requires a 'decl' symbol reference attribute");

    auto decl = symbolTable.lookupSymbolIn(getParentModule(*this), declAttr);
    if (!decl) return emitOpError("cannot resolve symbol '") << declAttr << "' to declaration";

    if (!mlir::isa<BMv2IR::HeaderInstanceOp>(decl))
        return emitOpError("invalid symbol reference: ") << decl << ", expected header instance";

    return mlir::success();
}

void SymToValueOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getDecl().getLeafReference());
}

void BMv2IRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.cpp.inc"  // NOLINT
        >();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.cpp.inc"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.cpp.inc"  // NOLINT
