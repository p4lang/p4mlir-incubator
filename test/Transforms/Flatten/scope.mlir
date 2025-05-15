// RUN: p4mlir-opt --p4hir-flatten-cfg %s | FileCheck %s

!bit32 = !p4hir.bit<32>

module {
  // Should properly print/parse scope with implicit empty yield.
  // CHECK-LABEL: implicit_yield
  p4hir.func @implicit_yield() {
    p4hir.scope {
    }
    // CHECK-NOT: p4hir.scope    
    // CHECK-NEXT: p4hir.return
    p4hir.return
  }

  // Should properly print/parse scope with explicit yield.
  // CHECK-LABEL: explicit_yield
  // CHECK-NOT: p4hir.scope    
  p4hir.func @explicit_yield() {
    %0 = p4hir.scope {
      %a = p4hir.variable ["a", init] : <!bit32>
      %1 = p4hir.read %a : <!bit32>
      p4hir.yield %1 : !bit32
    } : !bit32
    // CHECK: p4hir.br ^[[bb1:.*]]
    // CHECK: ^[[bb1]]:
    // CHECK: %[[val:.*]] = p4hir.read
    // CHECK: p4hir.br ^[[bb2:.*]](%[[val]] : !b32i)
    // CHECK: ^[[bb2]](%0: !b32i):
    // CHECK: p4hir.return
    p4hir.return
  }
}
