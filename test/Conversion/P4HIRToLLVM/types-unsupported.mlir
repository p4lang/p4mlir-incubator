// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// !p4hir.infint has no fixed width and thus no LLVM mapping. Ops using it
// must be left unconverted rather than crash.

!infint = !p4hir.infint

// CHECK-LABEL: module
module {
  // CHECK: p4hir.const
  %lhs = p4hir.const #p4hir.int<42> : !infint
  // CHECK: p4hir.const
  %rhs = p4hir.const #p4hir.int<1> : !infint
  // CHECK: p4hir.binop(add
  %add = p4hir.binop(add, %lhs, %rhs) : !infint
}

// -----

// P4 allows `bit<0>`, LLVM has no `i0`: such ops must be left alone as well.

!u0i = !p4hir.bit<0>

// CHECK-LABEL: module
module {
  // CHECK: p4hir.const
  %val = p4hir.const #p4hir.int<0> : !u0i
  // CHECK: p4hir.binop(add
  %add = p4hir.binop(add, %val, %val) : !u0i
}
