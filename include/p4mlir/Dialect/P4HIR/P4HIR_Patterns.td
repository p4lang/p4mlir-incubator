#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def IsZeroAttr : Constraint<CPred<"mlir::isa<P4::P4MLIR::P4HIR::IntAttr>($0) && "
                                 "mlir::cast<P4::P4MLIR::P4HIR::IntAttr>($0).getValue().isZero()">, "is zero">;

def IsOneAttr : Constraint<CPred<"mlir::isa<P4::P4MLIR::P4HIR::IntAttr>($0) && "
                                 "mlir::cast<P4::P4MLIR::P4HIR::IntAttr>($0).getValue().isOne()">, "is one">;

// binop(sub, 0, %x) ==> neg(%x)
def P4HIR_BinOp_SubZero : Pat<
  (BinOp BinOpKind_Sub, (ConstOp P4HIR_IntAttr:$value, $_, $_), $x),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(IsZeroAttr $value)]
>;

// binop(ssub, 0, %x) ==> neg(%x)
def P4HIR_BinOp_SubSatZero : Pat<
  (BinOp BinOpKind_SubSat, (ConstOp P4HIR_IntAttr:$value, $_, $_), $x),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(IsZeroAttr $value)]
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
