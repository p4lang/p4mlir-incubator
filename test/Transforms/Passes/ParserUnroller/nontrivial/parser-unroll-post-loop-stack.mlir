// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

!b1 = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int0_b1 = #p4hir.int<0> : !b1
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int5_b32i = #p4hir.int<5> : !b32i

// @post is outside the loop but accesses the loop's stack via .next. It is
// reached at index k+1 from iteration k, so it must be specialised per exit:
// @post (1), @post_1 (2), @post_2 (3); the exit from @parse_loop_3 is out of
// bounds.

// CHECK-LABEL: p4hir.state @parse_loop {
// CHECK:   } to @post{{$}}
// CHECK-LABEL: p4hir.state @parse_loop_1 {
// CHECK:   } to @post_1{{$}}
// CHECK-LABEL: p4hir.state @parse_loop_2 {
// CHECK:   } to @post_2{{$}}
// CHECK-LABEL: p4hir.state @parse_loop_3 {
// CHECK-NOT:   } to @post
// CHECK-LABEL: p4hir.state @post {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @post_1 {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @post_2 {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-NOT: @post_3
module @test_post_loop_stack {
    p4hir.parser @PostLoopStack(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        p4hir.state @parse_loop {
            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @post
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
        }
        p4hir.state @post {
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
