// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// LLVM has no `i0`, so `bit<0>` lowers to the empty struct: both have no bits
// and a single value.
//
// The inputs are derived from `p4mlir-translate --typeinference-only` output for the P4 code
// quoted before each action, with these declarations:
//
//   extern void sink0(in bit<0> x);
//   extern void sink8(in bit<8> x);
//   extern void sinkb(in bool x);
//   extern void out0(out bit<0> x);

// After `--canonicalize --p4hir-flatten-cfg --canonicalize`, a conditional
// expression passes `bit<0>` values, including the constant 0, as block
// arguments.
//
// action ternary(bool c, bit<0> a, bit<0> b) {
//     sink0(c ? a : b);
// }
//
// action ternary_cst(bool c, bit<0> a) {
//     sink0(c ? a : 0);
// }

!b0i = !p4hir.bit<0>
#in = #p4hir<dir in>
#int0_b0i = #p4hir.int<0> : !b0i

// CHECK-LABEL: module
module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sinkb(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "x"})

  // CHECK-LABEL: p4hir.func action @ternary(
  // CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !b0i to !llvm.struct<()>
  // CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !b0i to !llvm.struct<()>
  // CHECK-DAG:     %[[C:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !p4hir.bool to i1
  // CHECK:         llvm.cond_br %[[C]], ^[[JOIN:bb[0-9]+]](%[[A]] : !llvm.struct<()>), ^[[JOIN]](%[[B]] : !llvm.struct<()>)
  // CHECK-NEXT:  ^[[JOIN]](%{{.*}}: !llvm.struct<()>):
  p4hir.func action @ternary(%arg0: !p4hir.bool {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "c"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg2: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}) {
    p4hir.cond_br %arg0 ^bb1(%arg1 : !b0i), ^bb1(%arg2 : !b0i)
  ^bb1(%0: !b0i):  // 2 preds: ^bb0, ^bb0
    p4hir.call @sink0 (%0) : (!b0i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func action @ternary_cst(
  // CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !b0i to !llvm.struct<()>
  // CHECK-DAG:     %[[C:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !p4hir.bool to i1
  // CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.zero : !llvm.struct<()>
  // CHECK:         llvm.cond_br %[[C]], ^[[JOIN:bb[0-9]+]](%[[A]] : !llvm.struct<()>), ^[[JOIN]](%[[ZERO]] : !llvm.struct<()>)
  // CHECK-NEXT:  ^[[JOIN]](%{{.*}}: !llvm.struct<()>):
  p4hir.func action @ternary_cst(%arg0: !p4hir.bool {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "c"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) {
    %c0_b0i = p4hir.const #int0_b0i
    p4hir.cond_br %arg0 ^bb1(%arg1 : !b0i), ^bb1(%c0_b0i : !b0i)
  ^bb1(%0: !b0i):  // 2 preds: ^bb0, ^bb0
    p4hir.call @sink0 (%0) : (!b0i) -> ()
    p4hir.return
  }
}

// -----

// Without SROA and mem2reg, `bit<0>` variables reach the lowering. Folding
// their reads leaves most of them write-only, and canonicalization removes
// those. Variables passed by reference, e.g. as `out` arguments, remain. The
// empty struct already holds the only `bit<0>` value, so no default value is
// stored into them. The input is after `--p4hir-remove-soft-cf --canonicalize
// --p4hir-hoist-variables --p4hir-flatten-cfg --canonicalize`.
//
// action out_arg() {
//     bit<0> t;
//     out0(t);
//     sink0(t);
// }
//
// action scoped(bool c) {
//     if (c) {
//         bit<0> t;
//         out0(t);
//         sink0(t);
//     }
// }
//
// action inout_param(inout bit<0> v, bit<0> a) {
//     v = a;
// }

!b0i = !p4hir.bit<0>
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#int0_b0i = #p4hir.int<0> : !b0i

// CHECK-LABEL: module
module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @out0(!p4hir.ref<!b0i> {p4hir.dir = #out, p4hir.param_name = "x"})

  // CHECK-LABEL: p4hir.func action @out_arg(
  // CHECK:         %[[VAR:.*]] = llvm.alloca %{{.*}} x !llvm.struct<()> : (i64) -> !llvm.ptr
  // CHECK-NOT:     llvm.store
  // CHECK:         llvm.intr.lifetime.start %[[VAR]] : !llvm.ptr
  // CHECK-NEXT:    p4hir.call @out0
  // CHECK-NEXT:    llvm.intr.lifetime.end %[[VAR]] : !llvm.ptr
  p4hir.func action @out_arg() {
    %c0_b0i = p4hir.const #int0_b0i
    %x_out_arg = p4hir.variable ["x_out_arg"] : <!b0i>
    p4hir.lifetime_start %x_out_arg : <!b0i>
    p4hir.call @out0 (%x_out_arg) : (!p4hir.ref<!b0i>) -> ()
    p4hir.lifetime_end %x_out_arg : <!b0i>
    p4hir.call @sink0 (%c0_b0i) : (!b0i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func action @scoped(
  // CHECK:         %[[VAR:.*]] = llvm.alloca %{{.*}} x !llvm.struct<()> : (i64) -> !llvm.ptr
  // CHECK-NOT:     llvm.store
  // CHECK:         llvm.cond_br
  // CHECK-NEXT:  ^bb1:
  // CHECK-NEXT:    llvm.intr.lifetime.start %[[VAR]] : !llvm.ptr
  // CHECK-NEXT:    p4hir.call @out0
  // CHECK-NEXT:    llvm.intr.lifetime.end %[[VAR]] : !llvm.ptr
  p4hir.func action @scoped(%arg0: !p4hir.bool {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "c"}) {
    %c0_b0i = p4hir.const #int0_b0i
    %x_out_arg = p4hir.variable ["x_out_arg"] : <!b0i>
    p4hir.cond_br %arg0 ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    p4hir.lifetime_start %x_out_arg : <!b0i>
    p4hir.call @out0 (%x_out_arg) : (!p4hir.ref<!b0i>) -> ()
    p4hir.lifetime_end %x_out_arg : <!b0i>
    p4hir.call @sink0 (%c0_b0i) : (!b0i) -> ()
    p4hir.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func action @inout_param(
  // CHECK-DAG:     %[[REF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !p4hir.ref<!b0i> to !llvm.ptr
  // CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !b0i to !llvm.struct<()>
  // CHECK:         llvm.store %[[A]], %[[REF]] : !llvm.struct<()>, !llvm.ptr
  p4hir.func action @inout_param(%arg0: !p4hir.ref<!b0i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "v"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) {
    p4hir.assign %arg1, %arg0 : <!b0i>
    p4hir.return
  }
}

// -----

// These operations are expected to fold before lowering. If they remain,
// leave them unchanged: LLVM integer operations cannot accept empty structs.
//
// action unfolded(bit<0> a, bit<0> b, bit<8> x, int<8> y) {
//     sink0(a + b);
//     sink0(~a);
//     sinkb(a == b);
//     sink8(x ++ a);
//     sink8(a ++ y);
// }

!b0i = !p4hir.bit<0>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
#in = #p4hir<dir in>

// CHECK-LABEL: module
module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sink8(!b8i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func @sinkb(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "x"})

  // CHECK-LABEL: p4hir.func action @unfolded(
  // CHECK-NEXT:    p4hir.binop(add, %{{.*}}, %{{.*}}) : !b0i
  // CHECK:         p4hir.unary(cmpl, %{{.*}}) : !b0i
  // CHECK:         p4hir.cmp(eq, %{{.*}} : !b0i, %{{.*}} : !b0i)
  // CHECK:         p4hir.concat(%{{.*}} : !b8i, %{{.*}} : !b0i) : !b8i
  // CHECK:         p4hir.concat(%{{.*}} : !b0i, %{{.*}} : !i8i) : !b8i
  p4hir.func action @unfolded(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}, %arg2: !b8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}, %arg3: !i8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "y"}) {
    %add = p4hir.binop(add, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%add) : (!b0i) -> ()
    %cmpl = p4hir.unary(cmpl, %arg0) : !b0i
    p4hir.call @sink0 (%cmpl) : (!b0i) -> ()
    %eq = p4hir.cmp(eq, %arg0 : !b0i, %arg1 : !b0i)
    p4hir.call @sinkb (%eq) : (!p4hir.bool) -> ()
    %0 = p4hir.concat(%arg2 : !b8i, %arg0 : !b0i) : !b8i
    p4hir.call @sink8 (%0) : (!b8i) -> ()
    %1 = p4hir.concat(%arg0 : !b0i, %arg3 : !i8i) : !b8i
    p4hir.call @sink8 (%1) : (!b8i) -> ()
    p4hir.return
  }
}
