#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

#include "mlir/IR/Builders.h"
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
