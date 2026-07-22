// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!b1 = !p4hir.bit<1>
!b32 = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
#int0_b1 = #p4hir.int<0> : !b1

// No back-edge, yet @mid is reached at index 0 (straight from @start) and at
// index 1 (via @a's stack.next), so it specialises into two constant-index
// clones — loop-less unrolling, matching p4c.
// CHECK-LABEL: @test_loopless_merge
module @test_loopless_merge {
    p4hir.parser @LooplessMerge(%cond: !b1)() {
        // CHECK: p4hir.state @start
        // CHECK:   } to @a
        // CHECK:   } to @mid
        p4hir.state @start {
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @a
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @mid
            }
        }
        // CHECK: p4hir.state @a {
        // CHECK:   %[[CA:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[CA]]]
        // CHECK:   p4hir.transition to @mid_1
        p4hir.state @a {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32
            p4hir.transition to @mid
        }
        // CHECK: p4hir.state @mid {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @accept
        p4hir.state @mid {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32
            p4hir.transition to @accept
        }
        // CHECK: p4hir.state @mid_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @accept
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }
}
