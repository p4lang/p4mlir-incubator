// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
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
  // CHECK: llvm.intr.uadd.sat(%[[LHS]], %[[RHS]])
  %sadd = p4hir.binop(sadd, %lhs, %rhs) : !u32i
}

// -----

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !s32i
  // CHECK: llvm.intr.sadd.sat(%[[LHS]], %[[RHS]])
  %sadd = p4hir.binop(sadd, %lhs, %rhs) : !s32i
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
  // CHECK: llvm.intr.usub.sat(%[[LHS]], %[[RHS]])
  %ssub = p4hir.binop(ssub, %lhs, %rhs) : !u32i
}

// -----

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !s32i
  // CHECK: llvm.intr.ssub.sat(%[[LHS]], %[[RHS]])
  %ssub = p4hir.binop(ssub, %lhs, %rhs) : !s32i
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

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !u32i
  // CHECK: llvm.udiv %[[LHS]], %[[RHS]] : i32
  %div = p4hir.binop(div, %lhs, %rhs) : !u32i
}

// -----

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !s32i
  // CHECK: llvm.sdiv %[[LHS]], %[[RHS]] : i32
  %div = p4hir.binop(div, %lhs, %rhs) : !s32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !u32i
  // CHECK: llvm.urem %[[LHS]], %[[RHS]] : i32
  %mod = p4hir.binop(mod, %lhs, %rhs) : !u32i
}

// -----

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !s32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !s32i
  // CHECK: llvm.srem %[[LHS]], %[[RHS]] : i32
  %mod = p4hir.binop(mod, %lhs, %rhs) : !s32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.and %[[LHS]], %[[RHS]] : i32
  %and = p4hir.binop(and, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.or %[[LHS]], %[[RHS]] : i32
  %or = p4hir.binop(or, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
  %rhs = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: llvm.xor %[[LHS]], %[[RHS]] : i32
  %xor = p4hir.binop(xor, %lhs, %rhs) : !u32i
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
