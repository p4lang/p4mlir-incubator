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
  // CHECK-NEXT: }
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

// cast - widening that also changes the signedness extends by the source sign;
// such casts are produced by the canonicalizer composing cast chains.

!i8i = !p4hir.int<8>
!b8i = !p4hir.bit<8>
!i16i = !p4hir.int<16>
!b16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: %[[S:.*]] = llvm.mlir.constant(-1 : i8) : i8
  %s = p4hir.const #p4hir.int<-1> : !i8i
  // CHECK-NEXT: llvm.sext %[[S]] : i8 to i16
  %su = p4hir.cast(%s : !i8i) : !b16i
  // CHECK-NEXT: %[[U:.*]] = llvm.mlir.constant(-1 : i8) : i8
  %u = p4hir.const #p4hir.int<255> : !b8i
  // CHECK-NEXT: llvm.zext %[[U]] : i8 to i16
  %us = p4hir.cast(%u : !b8i) : !i16i
}

// -----

// cast - bit<1> to bool and bool to bit<1> is a no-op

!b1i = !p4hir.bit<1>

// CHECK-LABEL: module
module {
  // CHECK-NEXT: llvm.mlir.constant(true) : i1
  %one = p4hir.const #p4hir.int<1> : !b1i
  %flag = p4hir.cast(%one : !b1i) : !p4hir.bool
  // CHECK-NEXT: llvm.mlir.constant(false) : i1
  %false = p4hir.const #p4hir.bool<false> : !p4hir.bool
  // CHECK-NEXT: }
  %bit = p4hir.cast(%false : !p4hir.bool) : !b1i
}
