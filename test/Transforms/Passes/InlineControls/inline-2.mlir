// RUN: p4mlir-opt  --p4hir-inline-controls --canonicalize %s | FileCheck %s

// Fully inline empty controls and check properly instantiated extern object.

!b32i = !p4hir.bit<32>
#undir = #p4hir<dir undir>
#int10_b32i = #p4hir.int<10> : !b32i
// CHECK-LABEL: module
module {
  p4hir.extern @Y {
    p4hir.func @Y(!b32i {p4hir.dir = #undir, p4hir.param_name = "b"})
    p4hir.func @get() -> !b32i
  }
  // CHECK-LABEL: p4hir.control @Callee1
  p4hir.control @Callee1()() {
    // CHECK-DAG: %[[CONST_10:.*]] = p4hir.const #int10_b32i
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @ext
    %c10_b32i = p4hir.const #int10_b32i
    p4hir.instantiate @Y (%c10_b32i : !b32i) as @ext
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
    }
  }
  // CHECK-LABEL: p4hir.control @Callee2
  p4hir.control @Callee2()() {
    // CHECK-DAG: %[[CONST_10:.*]] = p4hir.const #int10_b32i
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c1.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2.ext
    p4hir.instantiate @Callee1 () as @c1
    p4hir.instantiate @Callee1 () as @c2
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
      p4hir.apply @Callee2::@c1() : () -> ()
      p4hir.apply @Callee2::@c2() : () -> ()
    }
  }
  // CHECK-LABEL: p4hir.control @Caller
  p4hir.control @Caller()() {
    // CHECK-DAG: %[[CONST_10:.*]] = p4hir.const #int10_b32i
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c1.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2a.c1.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2a.c2.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2b.c1.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2b.c2.ext
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
