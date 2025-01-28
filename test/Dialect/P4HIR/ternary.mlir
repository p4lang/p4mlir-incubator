// RUN: p4mlir-opt %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module {
  // No need to check stuff. If it parses, it's fine.
  // CHECK: module
  %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
  %1 = p4hir.ternary(%0, true {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
    p4hir.yield %29 : !p4hir.bool
  }, false {
    %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
    p4hir.yield %29 : !p4hir.bool
  }) : (!p4hir.bool) -> !p4hir.bool

  %2 = p4hir.ternary(%1, true {
    %29 = p4hir.const #p4hir.int<42> : !p4hir.int<32>
    p4hir.yield %29 : !p4hir.int<32>
  }, false {
    %29 = p4hir.const #p4hir.int<100500> : !p4hir.int<32>
    p4hir.yield %29 : !p4hir.int<32>
  }) : (!p4hir.bool) -> !p4hir.int<32>
}
