#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h.inc"

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H
