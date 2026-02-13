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

// The select reads counter after it is incremented in the same state, so
// iteration k selects on k+1. The exit to @x fires at k=1: @loop_1 must keep
// its edge to @x and there must be no @loop_2.

// CHECK-LABEL: p4hir.state @loop {
// CHECK:   } to @loop_1
// CHECK-LABEL: p4hir.state @loop_1 {
// CHECK-NOT: @loop_2
// CHECK:   } to @x
// CHECK-NOT: @loop_2
module @test_select_after_assign {
    p4hir.parser @SelectAfterAssign()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %zero = p4hir.const #int0_b32i
            p4hir.assign %zero, %counter : <!b32i>
            p4hir.transition to @loop
        }
        p4hir.state @loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            %counterVal = p4hir.read %counter : <!b32i>
            %one = p4hir.const #int1_b32i
            %newCounter = p4hir.binop(add, %counterVal, %one) : !b32i
            p4hir.assign %newCounter, %counter : <!b32i>
            %updatedVal = p4hir.read %counter : <!b32i>
            p4hir.transition_select %updatedVal : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int2_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
                } to @x
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @loop
            }
        }
        p4hir.state @x {
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
