// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
!arr_8xhdr = !p4hir.array<8x!hdr>
!hs8 = !p4hir.header_stack<8x!hdr>

#int4_b32i = #p4hir.int<4> : !b32i

// Loop accesses two stacks (hs4 size 4, hs8 size 8) and the select exits
// when nextIndex(stack1) + nextIndex(stack2) == 4. Both indices advance by 1
// each iteration, so the sum at iteration k is 2k. Exit fires at k=2 (sum=4).
// Stack OOB would be at k=4 (min of 4, 8). Select exit (k=2) is tighter.
// Expected: 3 clones (indices 0, 1, 2→accept), not 4.

// CHECK-LABEL: @test_hs_hs_sum_exit
module @test_hs_hs_sum_exit {
    // CHECK: p4hir.parser @HsHsSumParser
    p4hir.parser @HsHsSumParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @accept
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %stack1 = p4hir.variable ["stack1"] : <!hs4>
            %ni1_ref = p4hir.struct_field_ref %stack1["nextIndex"] : <!hs4>
            %ni1 = p4hir.read %ni1_ref : <!b32i>
            %d1_ref = p4hir.struct_field_ref %stack1["data"] : <!hs4>
            %e1_ref = p4hir.array_element_ref %d1_ref[%ni1] : !p4hir.ref<!arr_4xhdr>, !b32i

            %stack2 = p4hir.variable ["stack2"] : <!hs8>
            %ni2_ref = p4hir.struct_field_ref %stack2["nextIndex"] : <!hs8>
            %ni2 = p4hir.read %ni2_ref : <!b32i>
            %d2_ref = p4hir.struct_field_ref %stack2["data"] : <!hs8>
            %e2_ref = p4hir.array_element_ref %d2_ref[%ni2] : !p4hir.ref<!arr_8xhdr>, !b32i

            %sum = p4hir.binop(add, %ni1, %ni2) : !b32i
            p4hir.transition_select %sum : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int4_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
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
