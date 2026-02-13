// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s


!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_2xhdr = !p4hir.array<2x!hdr>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs2 = !p4hir.header_stack<2x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

// Two-state SCC where the same stack variable "stack" is declared as hs4 in
// @stateA and hs2 in @stateB. The smaller bound (2) should win, producing
// 2 clones (indices 0, 1→OOB), not 4.

// CHECK-LABEL: @test_conflicting_stack_sizes
module @test_conflicting_stack_sizes {
    // CHECK: p4hir.parser @ConflictingSizeParser
    p4hir.parser @ConflictingSizeParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.transition to @stateB
        // CHECK: p4hir.state @stateB {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.transition to @stateA_1
        // CHECK: p4hir.state @stateA_1 {
        // CHECK:   p4hir.array_element_ref
        // CHECK:   p4hir.transition to @start_outOfBound_0
        // CHECK-NOT: @stateB_1
        // expected-warning@below {{header stack '#0' appears with conflicting sizes in the same SCC; using the smaller bound}}
        p4hir.state @stateA {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.transition to @stateB
        }
        p4hir.state @stateB {
            %stack = p4hir.variable ["stack"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhdr>, !b32i
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
