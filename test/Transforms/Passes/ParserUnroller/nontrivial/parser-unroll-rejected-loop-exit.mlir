// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll="max-unroll-depth=2" | FileCheck %s

!b1 = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int0_b1 = #p4hir.int<0> : !b1

// @r is a loop over stack that exceeds max-unroll-depth and stays rolled.
// @s is reached both from @start (nextIndex 0) and from @r (any nextIndex).
// The edge from @r must not land in a copy of @s with a constant index.

// CHECK-LABEL: p4hir.state @r {
// CHECK:   } to @s{{$}}
// CHECK-LABEL: p4hir.state @s {
// CHECK-NOT:   p4hir.const
// CHECK:   p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs_4xhdr>
// CHECK-NEXT:   p4hir.transition to @accept
module @test_rejected_loop_exit {
    p4hir.parser @RejectedLoopExit(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @s
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @r
            }
        }
        p4hir.state @r {
            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @s
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @r
            }
        }
        p4hir.state @s {
            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>
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
