// RUN: p4mlir-opt %s | FileCheck %s
// Check for P4HIR syntax.

// CHECK-LABEL: test
module @test {

  %0 = p4hir.const 42 : ui32 -> !p4hir.bit<32>

}
