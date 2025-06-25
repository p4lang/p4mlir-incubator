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

// binop(mul, %x, 1) ==> %x
def BinOpMulOne : Pat<
  (BinOp BinOpKind_Mul, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsOneConst $rhs)]
>;


// binop(div, %x, 1) ==> %x
def BinOpDivOneRHS : Pat<
  (BinOp BinOpKind_Div, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsOneConst $rhs)]
>;

// binop(add, %x, 0) ==> %x
def BinOpAddZero : Pat<
  (BinOp BinOpKind_Add, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsZeroConst $rhs)]
>;

// binop(sadd, %x, 0) ==> %x
def BinOpAddSatZero : Pat<
  (BinOp BinOpKind_AddSat, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsZeroConst $rhs)]
>;

// binop(sub, 0, %rhs) ==> neg(%rhs)
def BinOpSubZeroLHS : Pat<
  (BinOp BinOpKind_Sub, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

// binop(sub, %x, 0) ==> %x
def BinOpSubZeroRHS : Pat<
  (BinOp BinOpKind_Sub, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsZeroConst $rhs)]
>;

// binop(ssub, 0, %x) ==> neg(%x)
def BinOpSubSatZeroLHS : Pat<
  (BinOp BinOpKind_SubSat, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

// binop(ssub, %x, 0) ==> %x
def BinOpSubSatZeroRHS : Pat<
  (BinOp BinOpKind_SubSat, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsZeroConst $rhs)]
>;

// binop(xor, %x, 0) ==> %x
def BinOpBXorZero : Pat<
  (BinOp BinOpKind_BXor, $lhs, $rhs),
  (replaceWithValue $lhs),
  [(IsZeroConst $rhs)]
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
