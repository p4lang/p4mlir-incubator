// RUN: p4mlir-opt --pass-pipeline='builtin.module(any(mem2reg))' < %s | FileCheck %s
!i32i = !p4hir.int<32>
!b9i = !p4hir.bit<9>
!PortId_t = !p4hir.struct<"PortId_t", _v: !b9i>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int100500_i32i = #p4hir.int<100500> : !i32i
#int42_i32i = #p4hir.int<42> : !i32i
module {
  // CHECK-LABEL: p4hir.func @ifthen
  p4hir.func @ifthen() {
  // CHECK-NOT: p4hir.variable
    %a = p4hir.variable ["a", init] : <!p4hir.bool>
    %false = p4hir.const #false
    p4hir.cond_br %false ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %true = p4hir.const #true
    p4hir.assign %true, %a : <!p4hir.bool>
    p4hir.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @ifthenelse
  p4hir.func @ifthenelse() -> !p4hir.bool {
    %a = p4hir.variable ["a", init] : <!p4hir.bool>
    %false = p4hir.const #false
    p4hir.cond_br %false ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %true = p4hir.const #true
    p4hir.assign %true, %a : <!p4hir.bool>
    p4hir.br ^bb3
  ^bb2:  // pred: ^bb0
    %false_0 = p4hir.const #false
    p4hir.assign %false_0, %a : <!p4hir.bool>
    p4hir.br ^bb3
  // CHECK: ^bb3(%[[VAL:.*]]: !p4hir.bool):
  // CHECK: p4hir.return %[[VAL]]
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %val = p4hir.read %a : <!p4hir.bool>
    p4hir.return %val : !p4hir.bool 
  }

  // CHECK-LABEL: p4hir.func @struct_field_ref
  // CHECK: p4hir.variable ["p1"
  p4hir.func @struct_field_ref() -> !b9i {
    %vv = p4hir.variable ["vv"] : <!b9i>
    %val = p4hir.read %vv : <!b9i>
    %struct_PortId_t = p4hir.struct (%val) : !PortId_t
    %p1 = p4hir.variable ["p1", init] : <!PortId_t>
    p4hir.assign %struct_PortId_t, %p1 : <!PortId_t>
    %field.ref = p4hir.struct_field_ref %p1 ["_v"] : <!PortId_t>
    %b = p4hir.read %field.ref : <!b9i>
    p4hir.return %b : !b9i
  }
}
