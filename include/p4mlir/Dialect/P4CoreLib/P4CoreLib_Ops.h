#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h.inc"

#endif  // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H
