// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt  --p4hir-inline-controls %s | FileCheck %s

// Fully inline empty controls nested in various ways.

// CHECK-LABEL: module
module @p4_main {
  // CHECK-LABEL: p4hir.control @Callee1
  p4hir.control @Callee1()() {
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
    }
  }
  // CHECK-LABEL: p4hir.control @Callee2
  p4hir.control @Callee2()() {
    // CHECK-NOT: p4hir.instantiate
    p4hir.instantiate @p4_main::@Callee1 () as @c
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
      p4hir.apply @c() : () -> ()
      p4hir.apply @c() : () -> ()
    }
  }
  // CHECK-LABEL: p4hir.control @Caller
  p4hir.control @Caller()() {
    // CHECK-NOT: p4hir.instantiate
    p4hir.instantiate @p4_main::@Callee1 () as @c1
    p4hir.instantiate @p4_main::@Callee2 () as @c2a
    p4hir.instantiate @p4_main::@Callee2 () as @c2b
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
      p4hir.apply @c1() : () -> ()
      p4hir.apply @c2a() : () -> ()
      p4hir.apply @c2b() : () -> ()
    }
  }
}
