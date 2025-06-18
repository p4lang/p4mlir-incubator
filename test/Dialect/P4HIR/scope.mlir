// RUN: p4mlir-opt %s | FileCheck %s

!bit32 = !p4hir.bit<32>

module {
  // Should properly print/parse scope with implicit empty yield.
  // CHECK-LABEL: implicit_yield
  p4hir.func @implicit_yield() {
    p4hir.scope {
    }
    // CHECK: p4hir.scope {
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return
    p4hir.return
  }

  // Should properly print/parse scope with explicit yield.
  // CHECK-LABEL: explicit_yield
  p4hir.func @explicit_yield() {
    %0 = p4hir.scope {
      %a = p4hir.variable ["a", init] : <!bit32>
      %1 = p4hir.read %a : <!bit32>
      p4hir.yield %1 : !bit32
    } : !bit32
    // CHECK: %0 = p4hir.scope {
    //          [...]
    // CHECK:   p4hir.yield %{{.*}} : !b32i
    // CHECK: } : !b32i
    p4hir.return
  }
}
