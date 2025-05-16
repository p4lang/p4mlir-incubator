#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def IsZero : Constraint<CPred<"mlir::cast<::P4::P4MLIR::P4HIR::IntAttr>($0).isNullValue()">, "is zero attr">;

// def P4HIR_BinOp_AddZero : Pat<
//   (BinOp BinOpKind_Add, (either (ConstOp P4HIR_IntAttr:$value, $_, $_), $x)),
//   (replaceWithValue $x),
//   [(IsZero $value)]
// >;

def P4HIR_BinOp_AddZero_LHS : Pat<
  (BinOp BinOpKind_Add, $x, (ConstOp P4HIR_IntAttr:$value, $_, $_)),
  (replaceWithValue $x),
  [(IsZero $value)]
>;

def P4HIR_BinOp_AddZero_RHS : Pat<
  (BinOp BinOpKind_Add, (ConstOp P4HIR_IntAttr:$value, $_, $_), $x),
  (replaceWithValue $x),
  [(IsZero $value)]
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
