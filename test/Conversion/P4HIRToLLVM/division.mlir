// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s


!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !u32i
  // CHECK-NEXT: llvm.udiv %[[LHS]], %[[RHS]] : i32
  %div = p4hir.binop(div, %lhs, %rhs) : !u32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(42 : i32) : i32
  %lhs = p4hir.const #p4hir.int<42> : !u32i
  // CHECK: %[[RHS:.*]] = llvm.mlir.constant(2 : i32) : i32
  %rhs = p4hir.const #p4hir.int<2> : !u32i
  // CHECK-NEXT: llvm.urem %[[LHS]], %[[RHS]] : i32
  %mod = p4hir.binop(mod, %lhs, %rhs) : !u32i
}
