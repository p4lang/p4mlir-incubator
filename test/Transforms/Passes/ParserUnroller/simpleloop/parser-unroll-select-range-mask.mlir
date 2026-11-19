// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int6_b32i = #p4hir.int<6> : !b32i

// Counter k equals the iteration number. The range case 2..3 must match at
// k=2, so the loop exits to @accept there instead of falling through to the
// default case until the stack runs out of bounds.

// CHECK-LABEL: @test_select_range_exit
module @test_select_range_exit {
    p4hir.parser @RangeParser()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %zero = p4hir.const #int0_b32i
            p4hir.assign %zero, %counter : <!b32i>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   } to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   } to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   } to @accept
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %counterVal = p4hir.read %counter : <!b32i>

            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>

            %one = p4hir.const #int1_b32i
            %newCounter = p4hir.binop(add, %counterVal, %one) : !b32i
            p4hir.assign %newCounter, %counter : <!b32i>

            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %low = p4hir.const #int2_b32i
                    %high = p4hir.const #int3_b32i
                    %range = p4hir.range(%low, %high) : !p4hir.set<!b32i>
                    p4hir.yield %range : !p4hir.set<!b32i>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
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

// Same as above with a mask: (k ^ 2) & 6 == 0 first holds at k=2.

// CHECK-LABEL: @test_select_mask_exit
module @test_select_mask_exit {
    p4hir.parser @MaskParser()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %zero = p4hir.const #int0_b32i
            p4hir.assign %zero, %counter : <!b32i>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   } to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   } to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   } to @accept
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %counterVal = p4hir.read %counter : <!b32i>

            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>

            %one = p4hir.const #int1_b32i
            %newCounter = p4hir.binop(add, %counterVal, %one) : !b32i
            p4hir.assign %newCounter, %counter : <!b32i>

            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %key = p4hir.const #int2_b32i
                    %maskValue = p4hir.const #int6_b32i
                    %mask = p4hir.mask(%key, %maskValue) : !p4hir.set<!b32i>
                    p4hir.yield %mask : !p4hir.set<!b32i>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
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
