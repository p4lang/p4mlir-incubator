// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// A slice shifts the lowest selected bit to position zero and truncates to
// the width of the selected range.

!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[VAL:.*]] = llvm.mlir.constant(258 : i32) : i32
  %val = p4hir.const #p4hir.int<258> : !b32i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.lshr %[[VAL]], %[[LOW]] : i32
  // CHECK-NEXT: llvm.trunc %[[SHIFTED]] : i32 to i8
  %slice = p4hir.slice %val[15 : 8] : !b32i -> !b8i
}

// -----

// A slice starting at bit zero needs no shift, only a truncation.

!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[VAL:.*]] = llvm.mlir.constant(258 : i32) : i32
  %val = p4hir.const #p4hir.int<258> : !b32i
  // CHECK-NEXT: llvm.trunc %[[VAL]] : i32 to i8
  %slice = p4hir.slice %val[7 : 0] : !b32i -> !b8i
}

// -----

// Slicing operates on the bit pattern, so the shift must be logical even for
// signed inputs.

!i32i = !p4hir.int<32>
!b8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[VAL:.*]] = llvm.mlir.constant(-1 : i32) : i32
  %val = p4hir.const #p4hir.int<-1> : !i32i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.lshr %[[VAL]], %[[LOW]] : i32
  // CHECK-NEXT: llvm.trunc %[[SHIFTED]] : i32 to i8
  %slice = p4hir.slice %val[15 : 8] : !i32i -> !b8i
}

// -----

// Extracting a single bit.

!b32i = !p4hir.bit<32>
!b1i = !p4hir.bit<1>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[VAL:.*]] = llvm.mlir.constant(8 : i32) : i32
  %val = p4hir.const #p4hir.int<8> : !b32i
  // CHECK-NEXT: %[[LOW:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT: %[[SHIFTED:.*]] = llvm.lshr %[[VAL]], %[[LOW]] : i32
  // CHECK-NEXT: llvm.trunc %[[SHIFTED]] : i32 to i1
  %slice = p4hir.slice %val[3 : 3] : !b32i -> !b1i
}

// -----

// A full-width slice only changes the P4 type's signedness, which LLVM integer
// types do not model.

!i8i = !p4hir.int<8>
!b8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: llvm.mlir.constant(-1 : i8) : i8
  %val = p4hir.const #p4hir.int<-1> : !i8i
  // CHECK-NOT: llvm.lshr
  // CHECK-NOT: llvm.trunc
  %slice = p4hir.slice %val[7 : 0] : !i8i -> !b8i
}
