// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file -verify-diagnostics

// A divisor known to be zero is diagnosed rather than lowered into undefined behaviour.
// p4c rejects such programs too, so this only fires on hand-written IR.

!u32i = !p4hir.bit<32>

module {
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  %rhs = p4hir.const #p4hir.int<0> : !u32i
  // expected-error @below {{division by zero}}
  %div = p4hir.binop(div, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

module {
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  %rhs = p4hir.const #p4hir.int<0> : !u32i
  // expected-error @below {{modulo by zero}}
  %mod = p4hir.binop(mod, %lhs, %rhs) : !u32i
}
