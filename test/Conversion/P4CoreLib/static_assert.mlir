// RUN: p4mlir-opt %s --lower-to-p4corelib --split-input-file -verify-diagnostics

// CASE 1: static_assert(true)
!string = !p4hir.string
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>

module {
  p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool annotations {corelib}

  p4hir.func @test_true_single() -> !p4hir.bool {
    %true = p4hir.const #true
    %result = p4hir.call @static_assert_1(%true) : (!p4hir.bool) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}

// -----

// CASE 2: static_assert(true, "message")
!string = !p4hir.string
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>

module {
  p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"},
                              !string {p4hir.dir = #undir, p4hir.param_name = "message"})
      -> !p4hir.bool annotations {corelib}

  p4hir.func @test_true_with_message() -> !p4hir.bool {
    %true = p4hir.const #true
    %msg = p4hir.const "version check" : !string
    %result = p4hir.call @static_assert_0(%true, %msg)
        : (!p4hir.bool, !string) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}

// -----

// ERROR CASE 1: static_assert(false)
#false = #p4hir.bool<false> : !p4hir.bool
#undir = #p4hir<dir undir>

module {
  p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"})
      -> !p4hir.bool annotations {corelib}

  p4hir.func @test_false() -> !p4hir.bool {
    %false = p4hir.const #false
    // expected-error @below {{static assertion failed}}
    // expected-error @below {{failed to legalize operation 'p4hir.call'}}
    %result = p4hir.call @static_assert_1(%false) : (!p4hir.bool) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}

// -----

// ERROR CASE 2: static_assert(false, "message")
!string = !p4hir.string
#false = #p4hir.bool<false> : !p4hir.bool
#undir = #p4hir<dir undir>

module {
  p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"},
                              !string {p4hir.dir = #undir, p4hir.param_name = "message"})
      -> !p4hir.bool annotations {corelib}

  p4hir.func @test_false_with_message() -> !p4hir.bool {
    %false = p4hir.const #false
    %msg = p4hir.const "this should fail" : !string
    // expected-error @below {{static assertion failed}}
    // expected-error @below {{failed to legalize operation 'p4hir.call'}}
    %result = p4hir.call @static_assert_0(%false, %msg)
        : (!p4hir.bool, !string) -> !p4hir.bool
    p4hir.return %result : !p4hir.bool
  }
}