// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b12i = !p4hir.bit<12>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[HIGH:.*]] = llvm.mlir.constant(3 : i4) : i4
  %high = p4hir.const #p4hir.int<3> : !b4i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(5 : i8) : i8
  %low = p4hir.const #p4hir.int<5> : !b8i
  // CHECK-NEXT: %[[HIGH_EXT:.*]] = llvm.zext %[[HIGH]] : i4 to i12
  // CHECK-NEXT: %[[LOW_EXT:.*]] = llvm.zext %[[LOW]] : i8 to i12
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(8 : i12) : i12
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.shl %[[HIGH_EXT]], %[[SHIFT]] : i12
  // CHECK-NEXT: llvm.or %[[SHIFTED]], %[[LOW_EXT]] : i12
  %concat = p4hir.concat(%high : !b4i, %low : !b8i) : !b12i
}

// -----

// Concatenation preserves the bit pattern of a signed operand rather than
// sign-extending it.

!i4i = !p4hir.int<4>
!b8i = !p4hir.bit<8>
!i12i = !p4hir.int<12>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[HIGH:.*]] = llvm.mlir.constant(-3 : i4) : i4
  %high = p4hir.const #p4hir.int<-3> : !i4i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(5 : i8) : i8
  %low = p4hir.const #p4hir.int<5> : !b8i
  // CHECK-NEXT: %[[HIGH_EXT:.*]] = llvm.zext %[[HIGH]] : i4 to i12
  // CHECK-NEXT: %[[LOW_EXT:.*]] = llvm.zext %[[LOW]] : i8 to i12
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(8 : i12) : i12
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.shl %[[HIGH_EXT]], %[[SHIFT]] : i12
  // CHECK-NEXT: llvm.or %[[SHIFTED]], %[[LOW_EXT]] : i12
  %concat = p4hir.concat(%high : !i4i, %low : !b8i) : !i12i
}

// -----

// The signed RHS is zero-extended so that sign extension does not affect the
// bits belonging to the LHS.

!b8i = !p4hir.bit<8>
!i4i = !p4hir.int<4>
!b12i = !p4hir.bit<12>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[HIGH:.*]] = llvm.mlir.constant(5 : i8) : i8
  %high = p4hir.const #p4hir.int<5> : !b8i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(-3 : i4) : i4
  %low = p4hir.const #p4hir.int<-3> : !i4i
  // CHECK-NEXT: %[[HIGH_EXT:.*]] = llvm.zext %[[HIGH]] : i8 to i12
  // CHECK-NEXT: %[[LOW_EXT:.*]] = llvm.zext %[[LOW]] : i4 to i12
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(4 : i12) : i12
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.shl %[[HIGH_EXT]], %[[SHIFT]] : i12
  // CHECK-NEXT: llvm.or %[[SHIFTED]], %[[LOW_EXT]] : i12
  %concat = p4hir.concat(%high : !b8i, %low : !i4i) : !b12i
}

// -----

// Single-bit operands are not special-cased.

!b1i = !p4hir.bit<1>
!b2i = !p4hir.bit<2>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[HIGH:.*]] = llvm.mlir.constant(true) : i1
  %high = p4hir.const #p4hir.int<1> : !b1i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(false) : i1
  %low = p4hir.const #p4hir.int<0> : !b1i
  // CHECK-NEXT: %[[HIGH_EXT:.*]] = llvm.zext %[[HIGH]] : i1 to i2
  // CHECK-NEXT: %[[LOW_EXT:.*]] = llvm.zext %[[LOW]] : i1 to i2
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(1 : i2) : i2
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.shl %[[HIGH_EXT]], %[[SHIFT]] : i2
  // CHECK-NEXT: llvm.or %[[SHIFTED]], %[[LOW_EXT]] : i2
  %concat = p4hir.concat(%high : !b1i, %low : !b1i) : !b2i
}
