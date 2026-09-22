// RUN: p4mlir-opt %s --p4hir-evaluate-static-assert | FileCheck %s

// Test: static_assert(true) is removed by EvaluateStaticAssertPass

#true = #p4hir.bool<true> : !p4hir.bool

module {
  // CHECK-LABEL: @test_true_removed
  // CHECK-NOT: p4corelib.static_assert
  // CHECK: p4hir.return
  p4hir.func @test_true_removed() -> !p4hir.bool {
    %true = p4hir.const #true
    %result = p4corelib.static_assert %true : !p4hir.bool -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }

  // Test: non-constant condition is left unchanged
  // CHECK-LABEL: @test_non_constant_unchanged
  // CHECK: p4corelib.static_assert
  p4hir.func @test_non_constant_unchanged(%arg0: !p4hir.bool) -> !p4hir.bool {
    %result = p4corelib.static_assert %arg0 : !p4hir.bool -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}

