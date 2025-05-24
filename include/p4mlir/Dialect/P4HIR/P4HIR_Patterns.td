#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def P4HIR_BinOp_AddZero : Pat<
  (BinOp BinOpKind_Add, (either (ConstOp P4HIR_IntAttr:$value, $_, $_), $x)),
  (replaceWithValue $x)
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
