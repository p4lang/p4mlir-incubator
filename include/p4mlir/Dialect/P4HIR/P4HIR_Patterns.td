#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def AddIntAttrs : NativeCodeCall<"addIntegerAttrs($_builder, $0, $1, $2)">;
def SubIntAttrs : NativeCodeCall<"subIntegerAttrs($_builder, $0, $1, $2)">;

def CreateConstOp : NativeCodeCall<"$_builder.create<P4HIR::ConstOp>($_loc, $0)">;

class HasValueConstraint<int value> : Constraint<
  CPred<"isIntegerValue($0, " # value # ")">,
  "operand has value " # value
>;

// add(add(x, c0), c1) -> add(x, c0 + c1)
def AddAddCst : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Add, $x, (ConstantLikeMatcher AnyAttr:$c0)),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (AddIntAttrs $res, $c0, $c1)))
>;

// sub(add(x, c0), c1) -> add(x, c0 - c1)
def SubAddCst : Pat<
  (BinOp:$res BinOpKind_Sub,
    (BinOp BinOpKind_Add, $x, (ConstantLikeMatcher AnyAttr:$c0)),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (SubIntAttrs $res, $c0, $c1)))
>;

// add(sub(x, c0), c1) -> add(x, c1 - c0)
def AddSubCst : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Sub, $x, (ConstantLikeMatcher AnyAttr:$c0)),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (SubIntAttrs $res, $c1, $c0)))
>;

// sub(sub(x, c0), c1) -> sub(x, c0 + c1)
def SubSubCst : Pat<
  (BinOp:$res BinOpKind_Sub,
    (BinOp BinOpKind_Sub, $x, (ConstantLikeMatcher AnyAttr:$c0)),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Sub, $x, (CreateConstOp (AddIntAttrs $res, $c0, $c1)))
>;

// add(sub(c0, x), c1) -> sub(c0 + c1, x)
def AddSubLhsCst : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Sub, (ConstantLikeMatcher AnyAttr:$c0), $x),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Sub, (CreateConstOp (AddIntAttrs $res, $c0, $c1)), $x)
>;

// sub(sub(c0, x), c1) -> sub(c0 - c1, x)
def SubSubLhsCst : Pat<
  (BinOp:$res BinOpKind_Sub,
    (BinOp BinOpKind_Sub, (ConstantLikeMatcher AnyAttr:$c0), $x),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Sub, (CreateConstOp (SubIntAttrs $res, $c0, $c1)), $x)
>;

// sub(c0, sub(x, c1)) -> sub(c0 + c1, x)
def SubRhsSubCst : Pat<
  (BinOp:$res BinOpKind_Sub,
    (ConstantLikeMatcher AnyAttr:$c0),
    (BinOp BinOpKind_Sub, $x, (ConstantLikeMatcher AnyAttr:$c1))),
  (BinOp BinOpKind_Sub, (CreateConstOp (AddIntAttrs $res, $c0, $c1)), $x)
>;

// sub(c0, sub(c1, x)) -> add(x, c0 - c1)
def SubRhsSubLhsCst : Pat<
  (BinOp:$res BinOpKind_Sub,
    (ConstantLikeMatcher AnyAttr:$c0),
    (BinOp BinOpKind_Sub, (ConstantLikeMatcher AnyAttr:$c1), $x)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (SubIntAttrs $res, $c0, $c1)))
>;

// add(neg(x), y) -> sub(y, x)
def AddNeg : Pat<
  (BinOp:$res BinOpKind_Add, (UnaryOp UnaryOpKind_Neg, $x), $y),
  (BinOp BinOpKind_Sub, $y, $x)
>;

// add(x, neg(y)) -> sub(x, y)
def AddRhsNeg : Pat<
  (BinOp:$res BinOpKind_Add, $x, (UnaryOp UnaryOpKind_Neg, $y)),
  (BinOp BinOpKind_Sub, $x, $y)
>;

// sub(x, neg(y)) -> add(x, y)
def SubRhsNeg : Pat<
  (BinOp:$res BinOpKind_Sub, $x, (UnaryOp UnaryOpKind_Neg, $y)),
  (BinOp BinOpKind_Add, $x, $y)
>;

// mul(x, -1) -> neg(x)
def MulToNeg : Pat<
  (BinOp:$res BinOpKind_Mul, $x, (ConstantLikeMatcher AnyAttr:$c0)),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(HasValueConstraint<-1> $c0)]
>;

// sub(0, x) -> neg(x)
def SubToNeg : Pat<
  (BinOp:$res BinOpKind_Sub, (ConstantLikeMatcher AnyAttr:$c0), $x),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(HasValueConstraint<0> $c0)]
>;

// sub_sat(0, x) -> neg(x)
def SubSatToNeg : Pat<
  (BinOp:$res BinOpKind_SubSat, (ConstantLikeMatcher AnyAttr:$c0), $x),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(HasValueConstraint<0> $c0)]
>;

// add(cmpl(x), 1) -> neg(x)
def AddCmplToNeg : Pat<
  (BinOp:$res BinOpKind_Add,
    (UnaryOp UnaryOpKind_Cmpl, $x),
    (ConstantLikeMatcher AnyAttr:$c0)),
  (UnaryOp UnaryOpKind_Neg, $x),
  [(HasValueConstraint<1> $c0)]
>;

// mul(mul(x, c0), c1) -> mul(x, c0 + c1)
def MulMulCst : Pat<
  (BinOp:$res BinOpKind_Mul,
    (BinOp BinOpKind_Mul, $x, (ConstantLikeMatcher AnyAttr:$c0)),
    (ConstantLikeMatcher AnyAttr:$c1)),
  (BinOp BinOpKind_Mul, $x, (CreateConstOp (AddIntAttrs $res, $c0, $c1)))
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
