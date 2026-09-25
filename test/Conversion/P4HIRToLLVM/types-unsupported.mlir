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
  // CHECK: p4hir.unary(minus
  %neg = p4hir.unary(minus, %lhs) : !infint
  // CHECK: p4hir.cmp(lt
  %lt = p4hir.cmp(lt, %lhs : !infint, %rhs : !infint)
  // CHECK: p4hir.shl
  %shl = p4hir.shl(%lhs, %rhs : !infint) : !infint
  // CHECK: p4hir.shr
  %shr = p4hir.shr(%lhs, %rhs : !infint) : !infint
  // CHECK: p4hir.slice
  %slice = p4hir.slice %lhs[3 : 0] : !infint -> !p4hir.bit<4>
}

// -----

// P4 allows `bit<0>`, LLVM has no `i0`: such ops must be left alone as well.

!u0i = !p4hir.bit<0>

// CHECK-LABEL: module
module {
  // CHECK: p4hir.const
  %lhs = p4hir.const #p4hir.int<0> : !u0i
  // CHECK: p4hir.const
  %rhs = p4hir.const #p4hir.int<0> : !u0i
  // CHECK: p4hir.binop(add
  %add = p4hir.binop(add, %lhs, %rhs) : !u0i
  // CHECK: p4hir.unary(cmpl
  %cmpl = p4hir.unary(cmpl, %lhs) : !u0i
  // CHECK: p4hir.cmp(eq
  %eq = p4hir.cmp(eq, %lhs : !u0i, %rhs : !u0i)
  // CHECK: p4hir.shl
  %shl = p4hir.shl(%lhs, %rhs : !u0i) : !u0i
  // CHECK: p4hir.shr
  %shr = p4hir.shr(%lhs, %rhs : !u0i) : !u0i
  // CHECK: p4hir.concat
  %concat = p4hir.concat(%lhs : !u0i, %rhs : !u0i) : !u0i
}

// -----

// A compile-time infint shift has no mapping.

!u32i = !p4hir.bit<32>
!infint = !p4hir.infint

// CHECK-LABEL: module
module {
  // CHECK: llvm.mlir.constant(1 : i32)
  %val = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: p4hir.const
  %shift = p4hir.const #p4hir.int<3> : !infint
  // CHECK: p4hir.shl
  %shl = p4hir.shl(%val, %shift : !infint) : !u32i
  // CHECK: p4hir.shr
  %shr = p4hir.shr(%val, %shift : !infint) : !u32i
}

// -----

// A `bit<0>` operand leaves the concatenation unconverted even though the
// result itself has an LLVM counterpart.

!u0i = !p4hir.bit<0>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK: p4hir.const
  %empty = p4hir.const #p4hir.int<0> : !u0i
  // CHECK: llvm.mlir.constant(5 : i8)
  %val = p4hir.const #p4hir.int<5> : !u8i
  // CHECK: p4hir.concat
  %concat = p4hir.concat(%empty : !u0i, %val : !u8i) : !u8i
}
