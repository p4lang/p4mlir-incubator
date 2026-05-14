// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// CHECK-LABEL: module
module @p4_main {
  p4hir.parser @callee1()() {
    p4hir.state @start {
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }
  p4hir.parser @callee2()() {
    p4hir.state @start {
      p4hir.transition to @reject
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @caller1
  p4hir.parser @caller1()() {
    p4hir.instantiate @p4_main::@callee1 () as @subparser

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @subparser() : () -> ()
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @caller2
  // Test with multiple calls in single state.
  p4hir.parser @caller2()() {
    p4hir.instantiate @p4_main::@callee1 () as @subparser1
    p4hir.instantiate @p4_main::@callee1 () as @subparser2
    p4hir.instantiate @p4_main::@callee1 () as @subparser3

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @subparser1() : () -> ()
      p4hir.apply @subparser2() : () -> ()
      p4hir.apply @subparser3() : () -> ()
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @caller3
  // Test with calls in different state.
  p4hir.parser @caller3()() {
    p4hir.instantiate @p4_main::@callee1 () as @subparser1
    p4hir.instantiate @p4_main::@callee1 () as @subparser2
    p4hir.instantiate @p4_main::@callee1 () as @subparser3

    // CHECK: p4hir.state @start
    // CHECK-NEXT: p4hir.parser_accept
    p4hir.state @start {
      p4hir.apply @subparser1() : () -> ()
      p4hir.transition to @next
    }
    p4hir.state @next {
      p4hir.apply @subparser2() : () -> ()
      p4hir.transition to @final
    }
    p4hir.state @final {
      p4hir.apply @subparser3() : () -> ()
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @caller4
  // Test subparser reject.
  p4hir.parser @caller4()() {
    p4hir.instantiate @p4_main::@callee2 () as @subparser
    p4hir.state @start {
      p4hir.apply @subparser() : () -> ()
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

}
