// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

// `bit<0>` has only one value: 0. Test folds for operations with bit<0>
// results and simplifications enabled by bit<0> operands.
//
// The input is `p4mlir-translate --typeinference-only` output for the P4 code
// quoted before each action, with these declarations:
//
//   extern void sink0(in bit<0> x);
//   extern void sink8(in bit<8> x);
//   extern void sinki8(in int<8> x);
//   extern void sinkb(in bool x);
//   header H { bit<0> z; bit<8> x; }
//   struct S { bit<0> z; bit<8> x; }

!b0i = !p4hir.bit<0>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!infint = !p4hir.infint
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
!H = !p4hir.header<"H", z: !b0i, x: !b8i, __valid: !validity_bit>
!S = !p4hir.struct<"S", z: !b0i, x: !b8i>
!arr_2xb0i = !p4hir.array<2x!b0i>
#int0_b0i = #p4hir.int<0> : !b0i
#int0_infint = #p4hir.int<0> : !infint

// CHECK-LABEL: module
module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sink8(!b8i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sinki8(!i8i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sinkb(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "x"})

  // action binop(bit<0> a, bit<0> b) {
  //     sink0(a + b);
  //     sink0(a - b);
  //     sink0(a * b);
  //     sink0(a |+| b);
  //     sink0(a |-| b);
  //     sink0(a & b);
  //     sink0(a | b);
  //     sink0(a ^ b);
  // }

  // CHECK-LABEL: p4hir.func action @binop(
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-COUNT-8:   p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @binop(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}) {
    %add = p4hir.binop(add, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%add) : (!b0i) -> ()
    %sub = p4hir.binop(sub, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%sub) : (!b0i) -> ()
    %mul = p4hir.binop(mul, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%mul) : (!b0i) -> ()
    %sadd = p4hir.binop(sadd, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%sadd) : (!b0i) -> ()
    %ssub = p4hir.binop(ssub, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%ssub) : (!b0i) -> ()
    %and = p4hir.binop(and, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%and) : (!b0i) -> ()
    %or = p4hir.binop(or, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%or) : (!b0i) -> ()
    %xor = p4hir.binop(xor, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%xor) : (!b0i) -> ()
    p4hir.return
  }

  // p4c truncates the literal 1 to the `bit<0>` value 0 with an overflow
  // warning. Matching it against the `x * 1` style rules must not try to
  // represent 1 in zero bits.
  //
  // action binop_cst(bit<0> a) {
  //     sink0(a * 1);
  // }

  // CHECK-LABEL: p4hir.func action @binop_cst(
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-NEXT:    p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @binop_cst(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) {
    %c0_b0i = p4hir.const #int0_b0i
    %mul = p4hir.binop(mul, %arg0, %c0_b0i) : !b0i
    p4hir.call @sink0 (%mul) : (!b0i) -> ()
    p4hir.return
  }

  // action unary(bit<0> a) {
  //     sink0(-a);
  //     sink0(~a);
  // }

  // CHECK-LABEL: p4hir.func action @unary(
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-COUNT-2:   p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @unary(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) {
    %minus = p4hir.unary(minus, %arg0) : !b0i
    p4hir.call @sink0 (%minus) : (!b0i) -> ()
    %cmpl = p4hir.unary(cmpl, %arg0) : !b0i
    p4hir.call @sink0 (%cmpl) : (!b0i) -> ()
    p4hir.return
  }

  // action cmp(bit<0> a, bit<0> b) {
  //     sinkb(a == b);
  //     sinkb(a != b);
  //     sinkb(a < b);
  //     sinkb(a <= b);
  //     sinkb(a > b);
  //     sinkb(a >= b);
  // }

  // CHECK-LABEL: p4hir.func action @cmp(
  // CHECK-DAG:     %[[FALSE:.*]] = p4hir.const #false
  // CHECK-DAG:     %[[TRUE:.*]] = p4hir.const #true
  // CHECK:         p4hir.call @sinkb (%[[TRUE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.call @sinkb (%[[FALSE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.call @sinkb (%[[FALSE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.call @sinkb (%[[TRUE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.call @sinkb (%[[FALSE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.call @sinkb (%[[TRUE]]) : (!p4hir.bool) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @cmp(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}) {
    %eq = p4hir.cmp(eq, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%eq) : (!p4hir.bool) -> ()
    %ne = p4hir.cmp(ne, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%ne) : (!p4hir.bool) -> ()
    %lt = p4hir.cmp(lt, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%lt) : (!p4hir.bool) -> ()
    %le = p4hir.cmp(le, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%le) : (!p4hir.bool) -> ()
    %gt = p4hir.cmp(gt, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%gt) : (!p4hir.bool) -> ()
    %ge = p4hir.cmp(ge, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%ge) : (!p4hir.bool) -> ()
    p4hir.return
  }

  // action cast(bit<0> a, bit<8> x) {
  //     sink8((bit<8>)a);
  //     sink0((bit<0>)x);
  // }

  // CHECK-LABEL: p4hir.func action @cast(
  // CHECK-DAG:     %[[C0_B0:.*]] = p4hir.const #int0_b0i
  // CHECK-DAG:     %[[C0_B8:.*]] = p4hir.const #int0_b8i
  // CHECK:         p4hir.call @sink8 (%[[C0_B8]]) : (!b8i) -> ()
  // CHECK-NEXT:    p4hir.call @sink0 (%[[C0_B0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @cast(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}) {
    %cast = p4hir.cast(%arg0 : !b0i) : !b8i
    p4hir.call @sink8 (%cast) : (!b8i) -> ()
    %cast_0 = p4hir.cast(%arg1 : !b8i) : !b0i
    p4hir.call @sink0 (%cast_0) : (!b0i) -> ()
    p4hir.return
  }

  // The result of a concatenation takes the signedness of its left-hand side,
  // so `a ++ y` needs a cast of `y` to the unsigned result.
  //
  // action concat(bit<0> a, bit<8> x, int<8> y) {
  //     sink0(a ++ a);
  //     sink8(x ++ a);
  //     sinki8(y ++ a);
  //     sink8(a ++ x);
  //     sink8(a ++ y);
  // }

  // CHECK-LABEL: p4hir.func action @concat(
  // CHECK-SAME:    %{{[^:]+}}: !b0i {{.*}}, %[[X:[^:]+]]: !b8i {{.*}}, %[[Y:[^:]+]]: !i8i
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-NEXT:    p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.call @sink8 (%[[X]]) : (!b8i) -> ()
  // CHECK-NEXT:    p4hir.call @sinki8 (%[[Y]]) : (!i8i) -> ()
  // CHECK-NEXT:    p4hir.call @sink8 (%[[X]]) : (!b8i) -> ()
  // CHECK-NEXT:    %[[CAST:.*]] = p4hir.cast(%[[Y]] : !i8i) : !b8i
  // CHECK-NEXT:    p4hir.call @sink8 (%[[CAST]]) : (!b8i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @concat(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}, %arg2: !i8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "y"}) {
    %0 = p4hir.concat(%arg0 : !b0i, %arg0 : !b0i) : !b0i
    p4hir.call @sink0 (%0) : (!b0i) -> ()
    %1 = p4hir.concat(%arg1 : !b8i, %arg0 : !b0i) : !b8i
    p4hir.call @sink8 (%1) : (!b8i) -> ()
    %2 = p4hir.concat(%arg2 : !i8i, %arg0 : !b0i) : !i8i
    p4hir.call @sinki8 (%2) : (!i8i) -> ()
    %3 = p4hir.concat(%arg0 : !b0i, %arg1 : !b8i) : !b8i
    p4hir.call @sink8 (%3) : (!b8i) -> ()
    %4 = p4hir.concat(%arg0 : !b0i, %arg2 : !i8i) : !b8i
    p4hir.call @sink8 (%4) : (!b8i) -> ()
    p4hir.return
  }

  // action extract(H h, S s, tuple<bit<0>, bit<8>> t, bit<32> i) {
  //     bit<0>[2] arr;
  //     sink0(h.z);
  //     sink0(s.z);
  //     sink0(t[0]);
  //     sink0(arr[i]);
  // }

  // CHECK-LABEL: p4hir.func action @extract(
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-COUNT-4:   p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @extract(%arg0: !H {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "h"}, %arg1: !S {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "s"}, %arg2: tuple<!b0i, !b8i> {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "t"}, %arg3: !b32i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "i"}) {
    %arr = p4hir.variable ["arr"] : <!arr_2xb0i>
    %z = p4hir.struct_extract %arg0["z"] : !H
    p4hir.call @sink0 (%z) : (!b0i) -> ()
    %z_0 = p4hir.struct_extract %arg1["z"] : !S
    p4hir.call @sink0 (%z_0) : (!b0i) -> ()
    %c0 = p4hir.const #int0_infint
    %t0 = p4hir.tuple_extract %arg2[0] : tuple<!b0i, !b8i>
    p4hir.call @sink0 (%t0) : (!b0i) -> ()
    %val = p4hir.read %arr : <!arr_2xb0i>
    %array_elt = p4hir.array_get %val[%arg3] : !arr_2xb0i, !b32i
    p4hir.call @sink0 (%array_elt) : (!b0i) -> ()
    p4hir.return
  }

  // action read(inout bit<0> v) {
  //     sink0(v);
  // }

  // CHECK-LABEL: p4hir.func action @read(
  // CHECK-NEXT:    %[[C0:.*]] = p4hir.const #int0_b0i
  // CHECK-NEXT:    p4hir.call @sink0 (%[[C0]]) : (!b0i) -> ()
  // CHECK-NEXT:    p4hir.return
  p4hir.func action @read(%arg0: !p4hir.ref<!b0i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "v"}) {
    %val = p4hir.read %arg0 : <!b0i>
    p4hir.call @sink0 (%val) : (!b0i) -> ()
    p4hir.return
  }
}
