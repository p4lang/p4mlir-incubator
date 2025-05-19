// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s

!error = !p4hir.error<NoError, SomeError>
#in = #p4hir<dir in>
#true = #p4hir.bool<true> : !p4hir.bool
#error_SomeError = #p4hir.error<SomeError, !error> : !error
module {
  // CHECK-NOT: @verify
  p4hir.func @verify(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, !error {p4hir.dir = #in, p4hir.param_name = "toSignal"}) annotations {corelib}
  // CHECK-LABEL: p2
  p4hir.parser @p2(%arg0: !p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, %arg1: !p4hir.ref<!p4hir.bool> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "matches"})() {
    p4hir.state @start {
      %true = p4hir.const #true
      %eq = p4hir.cmp(eq, %arg0, %true) : !p4hir.bool, !p4hir.bool
      %error_SomeError = p4hir.const #error_SomeError
      // CHECK: p4corelib.verify %eq signalling %error_SomeError : !error
      p4hir.call @verify (%eq, %error_SomeError) : (!p4hir.bool, !error) -> ()
      p4hir.transition to @p2::@next
    }
    p4hir.state @next {
      %true = p4hir.const #true
      p4hir.assign %true, %arg1 : <!p4hir.bool>
      p4hir.transition to @p2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p2::@start
  }
}
