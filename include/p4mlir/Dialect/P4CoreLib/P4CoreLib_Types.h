#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_H
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h.inc"

#endif  // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_H
