#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"

#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR::BMv2IR;

void BMv2IRDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.cpp.inc"  // NOLINT
        >();
}
