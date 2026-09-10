// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// All six kinds on unsigned bit<N> should pick the u* predicates.

!u32i = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  %a = p4hir.const #p4hir.int<1> : !u32i
  %b = p4hir.const #p4hir.int<2> : !u32i
  // CHECK: llvm.icmp "eq"
  %eq = p4hir.cmp(eq, %a : !u32i, %b : !u32i)
  // CHECK: llvm.icmp "ne"
  %ne = p4hir.cmp(ne, %a : !u32i, %b : !u32i)
  // CHECK: llvm.icmp "ult"
  %lt = p4hir.cmp(lt, %a : !u32i, %b : !u32i)
  // CHECK: llvm.icmp "ule"
  %le = p4hir.cmp(le, %a : !u32i, %b : !u32i)
  // CHECK: llvm.icmp "ugt"
  %gt = p4hir.cmp(gt, %a : !u32i, %b : !u32i)
  // CHECK: llvm.icmp "uge"
  %ge = p4hir.cmp(ge, %a : !u32i, %b : !u32i)
}

// -----

// All six kinds on signed int<N> should pick the s* predicates.

!i32i = !p4hir.int<32>

// CHECK-LABEL: module
module {
  %a = p4hir.const #p4hir.int<1> : !i32i
  %b = p4hir.const #p4hir.int<2> : !i32i
  // CHECK: llvm.icmp "eq"
  %eq = p4hir.cmp(eq, %a : !i32i, %b : !i32i)
  // CHECK: llvm.icmp "ne"
  %ne = p4hir.cmp(ne, %a : !i32i, %b : !i32i)
  // CHECK: llvm.icmp "slt"
  %lt = p4hir.cmp(lt, %a : !i32i, %b : !i32i)
  // CHECK: llvm.icmp "sle"
  %le = p4hir.cmp(le, %a : !i32i, %b : !i32i)
  // CHECK: llvm.icmp "sgt"
  %gt = p4hir.cmp(gt, %a : !i32i, %b : !i32i)
  // CHECK: llvm.icmp "sge"
  %ge = p4hir.cmp(ge, %a : !i32i, %b : !i32i)
}

// -----

// eq/ne on bool operands.

// CHECK-LABEL: module
module {
  %a = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %b = p4hir.const #p4hir.bool<false> : !p4hir.bool
  // CHECK: llvm.icmp "eq"
  %eq = p4hir.cmp(eq, %a : !p4hir.bool, %b : !p4hir.bool)
  // CHECK: llvm.icmp "ne"
  %ne = p4hir.cmp(ne, %a : !p4hir.bool, %b : !p4hir.bool)
}
