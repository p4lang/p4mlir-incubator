// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

// start -> A -+-> B -> A  (back-edge)
//             +-> C -> A  (back-edge)

!b1 = !p4hir.bit<1>

#int0_b1 = #p4hir.int<0> : !b1
#int1_b1 = #p4hir.int<1> : !b1

// CHECK-LABEL: @test_branching_loop
module @test_branching_loop {
    // CHECK: p4hir.parser @BranchingLoop
    p4hir.parser @BranchingLoop(%cond: !b1)() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA
        p4hir.state @stateA {
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @stateB
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @stateC
            }
        }
        // No header stack → not unrolled: both back-edges preserved, no clone.
        // CHECK: p4hir.state @stateB {
        // CHECK:   p4hir.transition to @stateA
        p4hir.state @stateB {
            p4hir.transition to @stateA
        }
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
