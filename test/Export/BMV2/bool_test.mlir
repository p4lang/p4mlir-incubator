// RUN: p4mlir-export-bmv2 --json-output %s | FileCheck %s

module {
  %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool
}

// CHECK: p4hir.const
