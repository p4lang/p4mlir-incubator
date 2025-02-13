#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H
