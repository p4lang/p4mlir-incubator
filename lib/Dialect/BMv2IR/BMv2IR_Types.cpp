#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"

#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"

using namespace mlir;
using namespace P4::P4MLIR;

void BMv2IR::BMv2IRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.cpp.inc"  // NOLINT
        >();
}
