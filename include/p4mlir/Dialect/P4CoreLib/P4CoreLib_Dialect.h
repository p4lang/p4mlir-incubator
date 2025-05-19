#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_H
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"

#endif  // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_H
