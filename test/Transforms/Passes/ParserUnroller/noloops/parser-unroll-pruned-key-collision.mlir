// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

!b1 = !p4hir.bit<1>
!b32i = !p4hir.bit<32>

#int0_b1 = #p4hir.int<0> : !b1
#int1_b32i = #p4hir.int<1> : !b32i

// @c always selects @x, so its edge to @y is pruned. @y is still reachable
// directly from @start with the same (empty) key: pruning must not mark it
// unreachable, and no StackOutOfBounds state may appear without stacks.

// CHECK-LABEL: p4hir.state @start {
// CHECK:   } to @c{{$}}
// CHECK:   } to @y{{$}}
// CHECK-NOT: outOfBound
module @test_pruned_key_collision {
    p4hir.parser @PrunedKeyCollision(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @c
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @y
            }
        }
        p4hir.state @c {
            %one = p4hir.const #int1_b32i
            p4hir.transition_select %one : !b32i {
                p4hir.select_case {
                    %c1 = p4hir.const #int1_b32i
                    %s1 = p4hir.set (%c1) : !p4hir.set<!b32i>
                    p4hir.yield %s1 : !p4hir.set<!b32i>
                } to @x
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @y
            }
        }
        p4hir.state @x {
            p4hir.transition to @accept
        }
        p4hir.state @y {
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
