// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// cast - sign change of the same width is a no-op

!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: llvm.mlir.constant(42 : i8) : i8
  %c = p4hir.const #p4hir.int<42> : !b8i
  // CHECK-NOT: llvm.zext
  // CHECK-NOT: llvm.sext
  // CHECK-NOT: llvm.trunc
  %s = p4hir.cast(%c : !b8i) : !i8i
}

// -----

// cast - unsigned value widened with zext

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(42 : i8) : i8
  %c = p4hir.const #p4hir.int<42> : !b8i
  // CHECK-NEXT: llvm.zext %[[C]] : i8 to i16
  %w = p4hir.cast(%c : !b8i) : !b16i
}

// -----

// cast - signed value widened with sext

!i8i = !p4hir.int<8>
!i16i = !p4hir.int<16>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i8) : i8
  %c = p4hir.const #p4hir.int<-1> : !i8i
  // CHECK-NEXT: llvm.sext %[[C]] : i8 to i16
  %w = p4hir.cast(%c : !i8i) : !i16i
}

// -----

// cast - value narrowed with trunc

!i16i = !p4hir.int<16>
!i8i = !p4hir.int<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i16) : i16
  %c = p4hir.const #p4hir.int<-1> : !i16i
  // CHECK-NEXT: llvm.trunc %[[C]] : i16 to i8
  %n = p4hir.cast(%c : !i16i) : !i8i
}

// -----

// cast - bool widened with zext

!b8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[B:.*]] = llvm.mlir.constant(true) : i1
  %b = p4hir.const #p4hir.bool<true> : !p4hir.bool
  // CHECK-NEXT: llvm.zext %[[B]] : i1 to i8
  %wide = p4hir.cast(%b : !p4hir.bool) : !b8i
}

// -----

// cast - bit<1> to bool is a no-op

!b1i = !p4hir.bit<1>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: llvm.mlir.constant(true) : i1
  %one = p4hir.const #p4hir.int<1> : !b1i
  // CHECK-NOT: llvm.trunc
  %flag = p4hir.cast(%one : !b1i) : !p4hir.bool
}

// -----

// cast - alias lowered as the type it wraps

!b9i = !p4hir.bit<9>
!b32i = !p4hir.bit<32>
!Narrow = !p4hir.alias<"Narrow", !b9i>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(10 : i9) : i9
  %c = p4hir.const #p4hir.int<10> : !b9i
  %alias = p4hir.cast(%c : !b9i) : !Narrow
  // CHECK-NEXT: llvm.zext %[[C]] : i9 to i32
  %wide = p4hir.cast(%alias : !Narrow) : !b32i
}
