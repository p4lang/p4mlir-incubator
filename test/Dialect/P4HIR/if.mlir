// RUN: p4mlir-opt %s | FileCheck %s

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
  p4hir.if %0 {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  }

  p4hir.if %0 {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  } else {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  }

  %1 = p4hir.if %0 -> !p4hir.bool {
    %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
    p4hir.yield %29 : !p4hir.bool
  } else {
    %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
    p4hir.yield %29 : !p4hir.bool
  }

  %2 = p4hir.if %1 -> !p4hir.int<32> {
    %29 = p4hir.const #p4hir.int<42> : !p4hir.int<32>
    p4hir.yield %29 : !p4hir.int<32>
  } else {
    %29 = p4hir.const #p4hir.int<100500> : !p4hir.int<32>
    p4hir.yield %29 : !p4hir.int<32>
  }
}
