#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"

#define GET_OP_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h.inc"

#endif  // P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H
