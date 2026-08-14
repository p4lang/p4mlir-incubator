// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


// CHECK-LABEL: @test_multistate_loop
module @test_multistate_loop {
    // CHECK: p4hir.parser @MyParser
    p4hir.parser @MyParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            // CHECK: p4hir.transition to @stateA
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA
        p4hir.state @stateA {
            p4hir.transition to @stateB
        }
        // CHECK: p4hir.state @stateB
        p4hir.state @stateB {
            p4hir.transition to @stateC
        }
        // No header stack -> not unrolled: A->B->C->A back-edge preserved, no clone.
        // CHECK: p4hir.state @stateC {
        // CHECK:   p4hir.transition to @stateA
        // CHECK-NOT: @stateA_1
        p4hir.state @stateC {
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
