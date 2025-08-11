// RUN: p4mlir-opt  --p4hir-inline-controls %s | FileCheck %s

// Fully inline empty controls nested in various ways.

// CHECK-LABEL: module
module {
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
    p4hir.instantiate @Callee1 () as @c
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
      p4hir.apply @Callee2::@c() : () -> ()
      p4hir.apply @Callee2::@c() : () -> ()
    }
  }
  // CHECK-LABEL: p4hir.control @Caller
  p4hir.control @Caller()() {
    // CHECK-NOT: p4hir.instantiate
    p4hir.instantiate @Callee1 () as @c1
    p4hir.instantiate @Callee2 () as @c2a
    p4hir.instantiate @Callee2 () as @c2b
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
      p4hir.apply @Caller::@c1() : () -> ()
      p4hir.apply @Caller::@c2a() : () -> ()
      p4hir.apply @Caller::@c2b() : () -> ()
    }
  }
}
