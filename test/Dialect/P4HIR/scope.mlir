// RUN: p4mlir-opt %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

!bit32 = !p4hir.bit<32>

module {
  // Should properly print/parse scope with implicit empty yield.
  // TODO: When functions will be ready, uncomment lines below
  // xHECK-LABEL: implicit_yield
  // p4hir.func @implicit_yield() {
    p4hir.scope {
    }
    // CHECK: p4hir.scope {
    // CHECK-NEXT: }
    // xHECK-NEXT: p4hir.return
    // p4hir.return
  // }

  // Should properly print/parse scope with explicit yield.
  // xHECK-LABEL: explicit_yield
  // p4hir.func @explicit_yield() {
    %0 = p4hir.scope {
      %1 = p4hir.alloca !bit32 ["a", init] : !p4hir.ref<!bit32>
      %2 = p4hir.load %1 : !p4hir.ref<!bit32>, !bit32
      p4hir.yield %2 : !bit32
    } : !bit32
    // CHECK: %0 = p4hir.scope {
    //          [...]
    // CHECK:   p4hir.yield %2 : !p4hir.bit<32>
    // CHECK: } : !p4hir.bit<32>
  //  p4hir.return
  //}

}
