// RUN: not p4mlir-opt %s --p4hir-evaluate-static-assert 2>&1 | FileCheck %s

// Test: static_assert(false) emits compile-time error

#false = #p4hir.bool<false> : !p4hir.bool

module {
  // CHECK: error: static assertion failed
  p4hir.func @test_false_error() -> !p4hir.bool {
    %false = p4hir.const #false
    %result = p4corelib.static_assert %false : !p4hir.bool -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}