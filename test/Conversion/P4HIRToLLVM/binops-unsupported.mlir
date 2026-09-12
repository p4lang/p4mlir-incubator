// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// P4 has no signed division or modulo: the spec defines `/` and `%` for non-negative
// arbitrary-precision integers only, and the frontend rejects signed operands with
// "Cannot operate on signed values". Such ops are left unconverted rather than lowered
// to `llvm.sdiv` / `llvm.srem`, whose semantics P4 does not define.

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  %rhs = p4hir.const #p4hir.int<2> : !s32i
  // CHECK: p4hir.binop(div
  %div = p4hir.binop(div, %lhs, %rhs) : !s32i
}

// -----

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  %rhs = p4hir.const #p4hir.int<2> : !s32i
  // CHECK: p4hir.binop(mod
  %mod = p4hir.binop(mod, %lhs, %rhs) : !s32i
}
