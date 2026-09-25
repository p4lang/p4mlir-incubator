// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: module
module {
  // CHECK: llvm.mlir.constant(true) : i1
  %t = p4hir.const #p4hir.bool<true> : !p4hir.bool
  // CHECK: llvm.mlir.constant(false) : i1
  %f = p4hir.const #p4hir.bool<false> : !p4hir.bool
}

// -----

!u8i = !p4hir.bit<8>
!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: llvm.mlir.constant(42 : i8) : i8
  %narrow = p4hir.const #p4hir.int<42> : !u8i
  // CHECK: llvm.mlir.constant(42 : i32) : i32
  %wide = p4hir.const #p4hir.int<42> : !u32i
}

// -----

// int<N> lowers to the same signless iN as bit<N>.

!s32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  // CHECK: llvm.mlir.constant(-1 : i32) : i32
  %neg = p4hir.const #p4hir.int<-1> : !s32i
}
