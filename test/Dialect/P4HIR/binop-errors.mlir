// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -split-input-file -verify-diagnostics

// P4 requires a nonzero denominator, but the only bit<0> value is 0. Like the
// p4c frontend, reject division and modulo of bit<0> values.
//
// The input is `p4mlir-translate --typeinference-only` output for the P4 code
// quoted before each action, with `extern void sink0(in bit<0> x);`.

!b0i = !p4hir.bit<0>
#in = #p4hir<dir in>

// action div(bit<0> a, bit<0> b) {
//     sink0(a / b);
// }

module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func action @div(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}) {
    // expected-error@+1 {{'p4hir.binop' op division by zero: the only value of '!p4hir.bit<0>' is 0}}
    %div = p4hir.binop(div, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%div) : (!b0i) -> ()
    p4hir.return
  }
}

// -----

!b0i = !p4hir.bit<0>
#in = #p4hir<dir in>

// action mod(bit<0> a, bit<0> b) {
//     sink0(a % b);
// }

module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func action @mod(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}, %arg1: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}) {
    // expected-error@+1 {{'p4hir.binop' op modulo by zero: the only value of '!p4hir.bit<0>' is 0}}
    %mod = p4hir.binop(mod, %arg0, %arg1) : !b0i
    p4hir.call @sink0 (%mod) : (!b0i) -> ()
    p4hir.return
  }
}

// -----

// p4c truncates the literal 1 to the bit<0> value 0, with an overflow warning.

!b0i = !p4hir.bit<0>
#in = #p4hir<dir in>
#int0_b0i = #p4hir.int<0> : !b0i

// action div_cst(bit<0> a) {
//     sink0(a / 1);
// }

module @p4_main {
  p4hir.func @sink0(!b0i {p4hir.dir = #in, p4hir.param_name = "x"})
  p4hir.func action @div_cst(%arg0: !b0i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) {
    %c0_b0i = p4hir.const #int0_b0i
    // expected-error@+1 {{'p4hir.binop' op division by zero: the only value of '!p4hir.bit<0>' is 0}}
    %div = p4hir.binop(div, %arg0, %c0_b0i) : !b0i
    p4hir.call @sink0 (%div) : (!b0i) -> ()
    p4hir.return
  }
}
