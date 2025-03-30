#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

namespace P4::P4MLIR::P4HIR {
void buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc);
}  // namespace  P4::P4MLIR::P4HIR

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_H
