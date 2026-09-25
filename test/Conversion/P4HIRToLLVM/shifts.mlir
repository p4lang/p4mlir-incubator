// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// shl - value and shift of the same width
!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i32) : i32
  %val = p4hir.const #p4hir.int<1> : !u32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(2 : i32) : i32
  %shift = p4hir.const #p4hir.int<2> : !u32i
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHIFT]] : i1, i32
  // CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i32
  %shl = p4hir.shl(%val, %shift : !u32i) : !u32i
}

// -----

// shl - value wider than the shift
!s32i = !p4hir.int<32>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(-1 : i32) : i32
  %val = p4hir.const #p4hir.int<-1> : !s32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(2 : i8) : i8
  %shift = p4hir.const #p4hir.int<2> : !u8i
  // CHECK-NEXT: %[[EXT:.*]] = llvm.zext %[[SHIFT]] : i8 to i32
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[EXT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[EXT]] : i1, i32
  // CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i32
  %shl = p4hir.shl(%val, %shift : !u8i) : !s32i
}

// -----

// shl - value narrower than the shift
!u8i = !p4hir.bit<8>
!u16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i8) : i8
  %val = p4hir.const #p4hir.int<1> : !u8i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(256 : i16) : i16
  %shift = p4hir.const #p4hir.int<256> : !u16i
  // CHECK-NEXT: %[[TRUNC:.*]] = llvm.trunc %[[SHIFT]] : i16 to i8
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(8 : i16) : i16
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i16
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[TRUNC]] : i1, i8
  // CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[VAL]], %[[SAFE_SHIFT]] : i8
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHL]] : i1, i8
  %shl = p4hir.shl(%val, %shift : !u16i) : !u8i
}

// -----

// shr for unsigned type - value and shift of the same width
!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(8 : i32) : i32
  %val = p4hir.const #p4hir.int<8> : !u32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(1 : i32) : i32
  %shift = p4hir.const #p4hir.int<1> : !u32i
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHIFT]] : i1, i32
  // CHECK-NEXT: %[[SHR:.*]] = llvm.lshr %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHR]] : i1, i32
  %shr = p4hir.shr(%val, %shift : !u32i) : !u32i
}

// -----

// shr for unsigned type - value wider than the shift
!u32i = !p4hir.bit<32>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(8 : i32) : i32
  %val = p4hir.const #p4hir.int<8> : !u32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(1 : i8) : i8
  %shift = p4hir.const #p4hir.int<1> : !u8i
  // CHECK-NEXT: %[[EXT:.*]] = llvm.zext %[[SHIFT]] : i8 to i32
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[EXT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[EXT]] : i1, i32
  // CHECK-NEXT: %[[SHR:.*]] = llvm.lshr %[[VAL]], %[[SAFE_SHIFT]] : i32
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHR]] : i1, i32
  %shr = p4hir.shr(%val, %shift : !u8i) : !u32i
}

// -----

// shr for unsigned type - value narrower than the shift
!u8i = !p4hir.bit<8>
!u16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(1 : i8) : i8
  %val = p4hir.const #p4hir.int<1> : !u8i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(256 : i16) : i16
  %shift = p4hir.const #p4hir.int<256> : !u16i
  // CHECK-NEXT: %[[TRUNC:.*]] = llvm.trunc %[[SHIFT]] : i16 to i8
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(8 : i16) : i16
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i16
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[ZERO]], %[[TRUNC]] : i1, i8
  // CHECK-NEXT: %[[SHR:.*]] = llvm.lshr %[[VAL]], %[[SAFE_SHIFT]] : i8
  // CHECK-NEXT: llvm.select %[[OVERFLOW]], %[[ZERO]], %[[SHR]] : i1, i8
  %shr = p4hir.shr(%val, %shift : !u16i) : !u8i
}

// -----

// shr for signed type - value and shift of the same width

!s32i = !p4hir.int<32>
!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(-1 : i32) : i32
  %val = p4hir.const #p4hir.int<-1> : !s32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(32 : i32) : i32
  %shift = p4hir.const #p4hir.int<32> : !u32i
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[MAX:.*]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[MAX]], %[[SHIFT]] : i1, i32
  // CHECK-NEXT: llvm.ashr %[[VAL]], %[[SAFE_SHIFT]] : i32
  %shr = p4hir.shr(%val, %shift : !u32i) : !s32i
}

// -----

// shr for signed type - value wider than the shift

!s32i = !p4hir.int<32>
!u8i = !p4hir.bit<8>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(-1 : i32) : i32
  %val = p4hir.const #p4hir.int<-1> : !s32i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(2 : i8) : i8
  %shift = p4hir.const #p4hir.int<2> : !u8i
  // CHECK-NEXT: %[[EXT:.*]] = llvm.zext %[[SHIFT]] : i8 to i32
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[EXT]], %[[WIDTH]] : i32
  // CHECK-NEXT: %[[MAX:.*]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[MAX]], %[[EXT]] : i1, i32
  // CHECK-NEXT: llvm.ashr %[[VAL]], %[[SAFE_SHIFT]] : i32
  %shr = p4hir.shr(%val, %shift : !u8i) : !s32i
}

// -----

// shr for signed type - value narrower than the shift

!s8i = !p4hir.int<8>
!u16i = !p4hir.bit<16>

// CHECK-LABEL: module
module {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(-1 : i8) : i8
  %val = p4hir.const #p4hir.int<-1> : !s8i
  // CHECK-NEXT: %[[SHIFT:.*]] = llvm.mlir.constant(256 : i16) : i16
  %shift = p4hir.const #p4hir.int<256> : !u16i
  // CHECK-NEXT: %[[TRUNC:.*]] = llvm.trunc %[[SHIFT]] : i16 to i8
  // CHECK-NEXT: %[[WIDTH:.*]] = llvm.mlir.constant(8 : i16) : i16
  // CHECK-NEXT: %[[OVERFLOW:.*]] = llvm.icmp "uge" %[[SHIFT]], %[[WIDTH]] : i16
  // CHECK-NEXT: %[[MAX:.*]] = llvm.mlir.constant(7 : i8) : i8
  // CHECK-NEXT: %[[SAFE_SHIFT:.*]] = llvm.select %[[OVERFLOW]], %[[MAX]], %[[TRUNC]] : i1, i8
  // CHECK-NEXT: llvm.ashr %[[VAL]], %[[SAFE_SHIFT]] : i8
  %shr = p4hir.shr(%val, %shift : !u16i) : !s8i
}
