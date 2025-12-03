#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_H
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <optional>

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/BuiltinAttributes.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_EnumAttrs.h.inc"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.h.inc"

#endif  // P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_H
