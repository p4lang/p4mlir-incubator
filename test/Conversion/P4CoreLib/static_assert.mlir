// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s

// Test lowering of static_assert calls to p4corelib.static_assert op

!string = !p4hir.string
#true = #p4hir.bool<true> : !p4hir.bool
#false = #p4hir.bool<false> : !p4hir.bool
#undir = #p4hir<dir undir>

module {
  p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool annotations {corelib}
  p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool annotations {corelib}

  // Test: static_assert(bool) - single argument form
  // CHECK-LABEL: @test_single_arg
  // CHECK: %[[C:.*]] = p4hir.const
  // CHECK: %[[R:.*]] = p4corelib.static_assert %[[C]]
  p4hir.func @test_single_arg() -> !p4hir.bool {
    %true = p4hir.const #true
    %result = p4hir.call @static_assert_1(%true) : (!p4hir.bool) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }

  // Test: static_assert(bool, string) - two argument form
  // CHECK-LABEL: @test_with_message
  // CHECK: %[[C:.*]] = p4hir.const
  // CHECK: %[[R:.*]] = p4corelib.static_assert %[[C]]
  p4hir.func @test_with_message() -> !p4hir.bool {
    %true = p4hir.const #true
    %msg = p4hir.const "compile-time check" : !string
    %result = p4hir.call @static_assert_0(%true, %msg) : (!p4hir.bool, !string) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }

  // Test: lowering works regardless of condition value
  // CHECK-LABEL: @test_false_lowering
  // CHECK: %[[C:.*]] = p4hir.const
  // CHECK: %[[R:.*]] = p4corelib.static_assert %[[C]]
  p4hir.func @test_false_lowering() -> !p4hir.bool {
    %false = p4hir.const #false
    %result = p4hir.call @static_assert_1(%false) : (!p4hir.bool) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}