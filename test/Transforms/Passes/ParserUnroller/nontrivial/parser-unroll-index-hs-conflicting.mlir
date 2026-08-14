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
#int2_b32i = #p4hir.int<2> : !b32i

// Loop accesses hs4 (size 4) but select on nextIndex exits at 2.
// The select-based exit (2) is tighter than the stack OOB (4), so only
// 3 clones should be produced (indices 0, 1, 2→accept), not 4.

// CHECK-LABEL: @test_select_exit_before_oob
module @test_select_exit_before_oob {
    // CHECK: p4hir.parser @SelectExitParser
    p4hir.parser @SelectExitParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   } to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   } to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK:   } to @accept
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.transition_select %nextIdx : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int2_b32i
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
