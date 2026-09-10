// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i32) : i32
  %x = p4hir.const #p4hir.int<5> : !u32i
  // plus is a no-op: no llvm op is inserted; the result reuses %[[X]].
  // CHECK-NOT: llvm.add
  // CHECK-NOT: llvm.sub
  %p = p4hir.unary(plus, %x) : !u32i
}

// -----

!i32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i32) : i32
  %x = p4hir.const #p4hir.int<5> : !i32i
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.sub %[[ZERO]], %[[X]] : i32
  %n = p4hir.unary(minus, %x) : !i32i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i32) : i32
  %x = p4hir.const #p4hir.int<5> : !u32i
  // CHECK: %[[ONES:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: llvm.xor %[[X]], %[[ONES]] : i32
  %c = p4hir.unary(cmpl, %x) : !u32i
}

// -----

// CHECK-LABEL: module
module {
  // CHECK: %[[B:.*]] = llvm.mlir.constant(true) : i1
  %b = p4hir.const #p4hir.bool<true> : !p4hir.bool
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK: llvm.xor %[[B]], %[[ONE]] : i1
  %n = p4hir.unary(not, %b) : !p4hir.bool
}
