// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// CHECK-LABEL: module
module {
  p4hir.parser @callee1()() {
    p4hir.state @start {
      p4hir.transition to @callee1::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee1::@start
  }
  p4hir.parser @callee2()() {
    p4hir.state @start {
      p4hir.transition to @callee2::@reject
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee2::@start
  }

  // CHECK-LABEL: p4hir.parser @caller1
  p4hir.parser @caller1()() {
    p4hir.instantiate @callee1 () as @subparser

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @caller1::@subparser() : () -> ()
      p4hir.transition to @caller1::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller1::@start
  }

  // CHECK-LABEL: p4hir.parser @caller2
  // Test with multiple calls in single state.
  p4hir.parser @caller2()() {
    p4hir.instantiate @callee1 () as @subparser1
    p4hir.instantiate @callee1 () as @subparser2
    p4hir.instantiate @callee1 () as @subparser3

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @caller2::@subparser1() : () -> ()
      p4hir.apply @caller2::@subparser2() : () -> ()
      p4hir.apply @caller2::@subparser3() : () -> ()
      p4hir.transition to @caller2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller2::@start
  }

  // CHECK-LABEL: p4hir.parser @caller3
  // Test with calls in different state.
  p4hir.parser @caller3()() {
    p4hir.instantiate @callee1 () as @subparser1
    p4hir.instantiate @callee1 () as @subparser2
    p4hir.instantiate @callee1 () as @subparser3

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @caller3::@subparser1() : () -> ()
      p4hir.transition to @caller3::@next
    }
    p4hir.state @next {
      p4hir.apply @caller3::@subparser2() : () -> ()
      p4hir.transition to @caller3::@final
    }
    p4hir.state @final {
      p4hir.apply @caller3::@subparser3() : () -> ()
      p4hir.transition to @caller3::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller3::@start
  }

  // CHECK-LABEL: p4hir.parser @caller4
  // Test subparser reject.
  p4hir.parser @caller4()() {
    p4hir.instantiate @callee2 () as @subparser
    p4hir.state @start {
      p4hir.apply @caller4::@subparser() : () -> ()
      p4hir.transition to @caller4::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller4::@start
  }

}
