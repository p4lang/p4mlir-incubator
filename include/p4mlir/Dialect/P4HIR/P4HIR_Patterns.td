#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def IsConst : Constraint<CPred<[{
  mlir::isa_and_present<::P4::P4MLIR::P4HIR::ConstOp>($0.getDefiningOp())
}]>>;

def IsNotConst : Constraint<CPred<[{
  !mlir::isa_and_present<::P4::P4MLIR::P4HIR::ConstOp>($0.getDefiningOp())
}]>>;

def IsZeroConst : Constraint<CPred<[{
  isZeroConst($0)
}]>>;

def IsOneConst : Constraint<CPred<[{
  isOneConst($0)
}]>>;

def IsCommutative : Constraint<CPred<[{
  ::P4::P4MLIR::P4HIR::BinOp::isCommutative(
    mlir::cast<::P4::P4MLIR::P4HIR::BinOpKindAttr>($0).getValue()
  )
}]>>;

// binop(commutative, c, %x) ==> binop(commutative, %x, c)
def BinOpCommutativeCanonicalization : Pat<
  (BinOp $kind, $lhs, $rhs),
  (BinOp $kind, $rhs, $lhs),
  [(IsCommutative $kind), (IsConst $lhs), (IsNotConst $rhs)]
>;

// binop(sub, 0, %rhs) ==> neg(%rhs)
def BinOpSubZero : Pat<
  (BinOp BinOpKind_Sub, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

// binop(ssub, 0, %x) ==> neg(%x)
def BinOpSubSatZero : Pat<
  (BinOp BinOpKind_SubSat, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
