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

// @b accesses data[nextIndex - 1] (hs.last) through a reference. That is not
// a .next access and must not advance the index: the loop {@a, @b} unrolls
// to 4 iterations with @a at indices 0..3.

// CHECK-LABEL: p4hir.state @a {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @a_1 {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @a_2 {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @a_3 {
// CHECK:   p4hir.header_stack_extract_next
// CHECK-LABEL: p4hir.state @b_3 {
// CHECK:   } to @accept
// CHECK:   } to @start_outOfBound_0
module @test_last_ref {
    p4hir.parser @LastRef(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition to @a
        }
        p4hir.state @a {
            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>
            p4hir.transition to @b
        }
        p4hir.state @b {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %one = p4hir.const #int1_b32i
            %lastIdx = p4hir.binop(sub, %nextIdx, %one) : !b32i
            %last_ref = p4hir.array_element_ref %data_ref[%lastIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @a
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
