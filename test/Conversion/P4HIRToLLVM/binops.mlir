// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.add %[[LHS]], %[[RHS]] : i32
  %add = p4hir.binop(add, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.sub %[[LHS]], %[[RHS]] : i32
  %sub = p4hir.binop(sub, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.mul %[[LHS]], %[[RHS]] : i32
  %mul = p4hir.binop(mul, %lhs, %rhs) : !u32i
}

// -----

// Composite check that (a + b) * c - d lowers to a chain of llvm ops

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[A:.*]] = llvm.mlir.constant(2 : i32) : i32
  %a = p4hir.const #p4hir.int<2> : !u32i
  // CHECK: %[[B:.*]] = llvm.mlir.constant(3 : i32) : i32
  %b = p4hir.const #p4hir.int<3> : !u32i
  // CHECK: %[[C:.*]] = llvm.mlir.constant(4 : i32) : i32
  %c = p4hir.const #p4hir.int<4> : !u32i
  // CHECK: %[[D:.*]] = llvm.mlir.constant(5 : i32) : i32
  %d = p4hir.const #p4hir.int<5> : !u32i

  // CHECK: %[[SUM:.*]] = llvm.add %[[A]], %[[B]] : i32
  %sum = p4hir.binop(add, %a, %b) : !u32i
  // CHECK: %[[PROD:.*]] = llvm.mul %[[SUM]], %[[C]] : i32
  %prod = p4hir.binop(mul, %sum, %c) : !u32i
  // CHECK: llvm.sub %[[PROD]], %[[D]] : i32
  %res = p4hir.binop(sub, %prod, %d) : !u32i
}

// -----

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
