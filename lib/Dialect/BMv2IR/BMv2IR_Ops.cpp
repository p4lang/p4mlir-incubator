#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"

using namespace P4::P4MLIR;

void BMv2IR::BMv2IRDialect::initialize() {
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
