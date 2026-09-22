// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i32) : i32
  %val = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: %[[SHIFT:.*]] = llvm.mlir.constant(2 : i32) : i32
  %shift = p4hir.const #p4hir.int<2> : !u32i
  // CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHIFT]] : i1, i32
  // CHECK: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i32
  %shl = p4hir.shl(%val, %shift : !u32i) : !u32i
}

// -----

!u32i = !p4hir.bit<32>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i32) : i32
  %val = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: %[[SHIFT:.*]] = llvm.mlir.constant(2 : i8) : i8
  %shift = p4hir.const #p4hir.int<2> : !u8i
  // CHECK: %[[EXT:.*]] = llvm.zext %[[SHIFT]] : i8 to i32
  // CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[EXT]], %[[WIDTH]] : i32
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[EXT]] : i1, i32
  // CHECK: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i32
  %shl = p4hir.shl(%val, %shift : !u8i) : !u32i
}

// -----

!u8i = !p4hir.bit<8>
!u16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i8) : i8
  %val = p4hir.const #p4hir.int<1> : !u8i
  // CHECK: %[[SHIFT:.*]] = llvm.mlir.constant(256 : i16) : i16
  %shift = p4hir.const #p4hir.int<256> : !u16i
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[SHIFT]] : i16 to i8
  // CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(8 : i16) : i16
  // CHECK: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i16
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[TRUNC]] : i1, i8
  // CHECK: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i8
  // CHECK: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i8
  %shl = p4hir.shl(%val, %shift : !u16i) : !u8i
}

// -----

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(8 : i32) : i32
  %val = p4hir.const #p4hir.int<8> : !u32i
  // CHECK: %[[SHIFT:.*]] = llvm.mlir.constant(1 : i32) : i32
  %shift = p4hir.const #p4hir.int<1> : !u32i
  // CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHIFT]] : i1, i32
  // CHECK: %[[SHR:.*]] = llvm.lshr %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHR]] : i1, i32
  %shr = p4hir.shr(%val, %shift : !u32i) : !u32i
}

// -----

!s32i = !p4hir.int<32>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  %val = p4hir.const #p4hir.int<-1> : !s32i
  %shift = p4hir.const #p4hir.int<1> : !u8i
  // CHECK: llvm.shl
  %shl = p4hir.shl(%val, %shift : !u8i) : !s32i
}

// -----

!s32i = !p4hir.int<32>
!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(-1 : i32) : i32
  %val = p4hir.const #p4hir.int<-1> : !s32i
  // CHECK: %[[SHIFT:.*]] = llvm.mlir.constant(32 : i32) : i32
  %shift = p4hir.const #p4hir.int<32> : !u32i
  // CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK: %[[MAX:.*]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[MAX]], %[[SHIFT]] : i1, i32
  // CHECK: llvm.ashr %[[VAL]], %[[SAFE_SHIFT]] : i32
  %shr = p4hir.shr(%val, %shift : !u32i) : !s32i
}
