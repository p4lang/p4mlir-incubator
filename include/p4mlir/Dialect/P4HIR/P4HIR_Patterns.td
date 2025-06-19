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
  [&]() {
    if (auto constOp = mlir::cast_if_present<::P4::P4MLIR::P4HIR::ConstOp>($0.getDefiningOp())) {
      if (auto intAttr = mlir::dyn_cast<::P4::P4MLIR::P4HIR::IntAttr>(constOp.getValue())) {
        return intAttr.getValue().isZero();
      }
    }
    return false;
  }()
}]>>;

def IsOneConst : Constraint<CPred<[{
  [&]() {
    if (auto constOp = mlir::cast_if_present<::P4::P4MLIR::P4HIR::ConstOp>($0.getDefiningOp())) {
      if (auto intAttr = mlir::dyn_cast<::P4::P4MLIR::P4HIR::IntAttr>(constOp.getValue())) {
        return intAttr.getValue().isOne();
      }
    }
    return false;
  }()
}]>>;

// binop(sub, 0, %rhs) ==> neg(%rhs)
def P4HIR_BinOp_SubZero : Pat<
  (BinOp BinOpKind_Sub, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

// binop(ssub, 0, %x) ==> neg(%x)
def P4HIR_BinOp_SubSatZero : Pat<
  (BinOp BinOpKind_SubSat, $lhs, $rhs),
  (UnaryOp UnaryOpKind_Neg, $rhs),
  [(IsZeroConst $lhs)]
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
