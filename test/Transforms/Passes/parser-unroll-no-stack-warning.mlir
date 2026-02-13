// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s


module @test_no_stack_warning {
    p4hir.parser @NoStackParser()() {
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // No header stack → not unrolled: self-loop kept, no clone.
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.transition to @parse_loop
        // CHECK-NOT: @parse_loop_1
        // expected-warning@below {{parser loop at state 'parse_loop' has no header stack operations; cannot infer unroll depth}}
        p4hir.state @parse_loop {
            p4hir.transition to @parse_loop
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

// -----


module @test_no_stack_multistate {
    p4hir.parser @NoStackMulti()() {
        p4hir.state @start {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA {
        // CHECK:   p4hir.transition to @stateB
        // expected-warning@below {{parser loop at state 'stateA' has no header stack operations; cannot infer unroll depth}}
        p4hir.state @stateA {
            p4hir.transition to @stateB
        }
        // CHECK: p4hir.state @stateB {
        // CHECK:   p4hir.transition to @stateA
        // CHECK-NOT: @stateA_1
        p4hir.state @stateB {
            p4hir.transition to @stateA
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
