// RUN: p4mlir-opt %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module {
  // No need to check stuff. If it parses, it's fine.
  // CHECK: module
  %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
  p4hir.if %0 {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  }

  p4hir.if %0 {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  } else {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  }
}
