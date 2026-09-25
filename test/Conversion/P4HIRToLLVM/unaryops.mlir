// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

!u128i = !p4hir.bit<128>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i128) : i128
  %x = p4hir.const #p4hir.int<5> : !u128i
  // plus is a no-op: no llvm op is inserted; the result reuses %[[X]].
  // CHECK-NOT: llvm.add
  // CHECK-NOT: llvm.sub
  %p = p4hir.unary(plus, %x) : !u128i
}

// -----

!i128i = !p4hir.int<128>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i128) : i128
  %x = p4hir.const #p4hir.int<5> : !i128i
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i128) : i128
  // CHECK: llvm.sub %[[ZERO]], %[[X]] : i128
  %n = p4hir.unary(minus, %x) : !i128i
}

// -----

!u128i = !p4hir.bit<128>

// CHECK-LABEL: module
module {
  // CHECK: %[[X:.*]] = llvm.mlir.constant(5 : i128) : i128
  %x = p4hir.const #p4hir.int<5> : !u128i
  // CHECK: %[[ONES:.*]] = llvm.mlir.constant(-1 : i128) : i128
  // CHECK: llvm.xor %[[X]], %[[ONES]] : i128
  %c = p4hir.unary(cmpl, %x) : !u128i
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
