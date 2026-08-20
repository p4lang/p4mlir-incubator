// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(any(mem2reg))' < %s | FileCheck %s
!i32i = !p4hir.int<32>
!b9i = !p4hir.bit<9>
!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!PortId_t = !p4hir.struct<"PortId_t", _v: !b9i>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int100500_i32i = #p4hir.int<100500> : !i32i
#int42_i32i = #p4hir.int<42> : !i32i
#int0_b16i = #p4hir.int<0> : !b16i
#int171_b8i = #p4hir.int<171> : !b8i
#int205_b8i = #p4hir.int<205> : !b8i
// CHECK: #[[$ATTR_0:.+]] = #p4hir.bool<false> : !p4hir.bool
// CHECK: #[[$ATTR_1:.+]] = #p4hir.bool<true> : !p4hir.bool
// CHECK: #[[$ATTR_205:.+]] = #p4hir.int<205> : !b8i
// CHECK: #[[$ATTR_171:.+]] = #p4hir.int<171> : !b8i
// CHECK: #[[$ATTR_9w0:.+]] = #p4hir.int<0> : !b16i
// CHECK: #[[$ATTR_5:.+]] = #p4hir.int<0> : !b9i
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

  // CHECK-LABEL:   p4hir.func @slice_var()
  // CHECK:           %[[VAL_0:.*]] = p4hir.const #[[$ATTR_9w0]]
  // CHECK:           %[[VAL_1:.*]] = p4hir.const #[[$ATTR_171]]
  // CHECK:           %[[VAL_2:.*]] = p4hir.slice %[[VAL_0]][15 : 8] : !b16i -> !b8i
  // CHECK:           %[[VAL_3:.*]] = p4hir.concat(%[[VAL_2]] : !b8i, %[[VAL_1]] : !b8i) : !b16i
  // CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_205]]
  // CHECK:           %[[VAL_5:.*]] = p4hir.slice %[[VAL_3]][7 : 0] : !b16i -> !b8i
  // CHECK:           %[[VAL_6:.*]] = p4hir.concat(%[[VAL_4]] : !b8i, %[[VAL_5]] : !b8i) : !b16i
  // CHECK:           %[[VAL_7:.*]] = p4hir.slice %[[VAL_6]][7 : 0] : !b16i -> !b8i
  // CHECK:           p4hir.return %[[VAL_7]]
  // CHECK:         }
  p4hir.func @slice_var() -> !b8i {
    %v = p4hir.variable ["v", init] : <!b16i>
    %c0 = p4hir.const #int0_b16i
    p4hir.assign %c0, %v : <!b16i>
    %cab = p4hir.const #int171_b8i
    p4hir.assign_slice %cab, %v[7 : 0] : !b8i -> <!b16i>
    %ccd = p4hir.const #int205_b8i
    p4hir.assign_slice %ccd, %v[15 : 8] : !b8i -> <!b16i>
    %r = p4hir.read_slice %v[7 : 0] : <!b16i> -> !b8i
    p4hir.return %r : !b8i
  }
}
