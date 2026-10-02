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
#int4_b32i = #p4hir.int<4> : !b32i

// Loop accesses hs4 (size 4) and maintains a counter variable.
// The select exits when nextIndex + counter == 4.
// Counter starts at 0 (initialized in @start) and increments each iteration.
// At iteration k: nextIndex=k, counter=k, sum=2k. Exit at k=2 (sum=4).
// Stack OOB would be at k=4. Select exit (k=2) is tighter.
// Expected: 3 clones (indices 0, 1, 2→accept), not 4.

// CHECK-LABEL: @test_counter_plus_index_exit
module @test_counter_plus_index_exit {
    // CHECK: p4hir.parser @CounterIndexParser
    p4hir.parser @CounterIndexParser()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %zero = p4hir.const #int0_b32i
            p4hir.assign %zero, %counter : <!b32i>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   } to @accept
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %counterVal = p4hir.read %counter : <!b32i>

            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i

            %sum = p4hir.binop(add, %nextIdx, %counterVal) : !b32i

            %one = p4hir.const #int1_b32i
            %newCounter = p4hir.binop(add, %counterVal, %one) : !b32i
            p4hir.assign %newCounter, %counter : <!b32i>

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
