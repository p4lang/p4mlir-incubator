#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_OpInterfaces.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#define GET_OP_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h.inc"

namespace P4::P4MLIR::BMv2IR {

constexpr llvm::StringRef standardMetadataOldStructName = "standard_metadata_t";
constexpr llvm::StringRef standardMetadataNewStructName = "standard_metadata";

mlir::StringAttr getUniqueNameInParentModule(mlir::Operation *op, llvm::Twine base);

llvm::FailureOr<bool> isTopLevelControl(P4HIR::ControlOp controlOp);

llvm::FailureOr<bool> isDeparserControl(P4HIR::ControlOp controlOp);

llvm::FailureOr<bool> isCalculationControl(P4HIR::ControlOp controlOp);

// Returns true if the if op is checking the result of a table_apply for hit or miss
bool isHitOrMissIf(mlir::Operation *op);

}  // namespace P4::P4MLIR::BMv2IR

#endif  // P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_H
