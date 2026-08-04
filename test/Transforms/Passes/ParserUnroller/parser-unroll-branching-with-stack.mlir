// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

//   start → stateA (extract hs2) → select → stateB → stateA  (back-edge)
//                                           → stateC → stateA  (back-edge)
//   stateA   → select → stateB, stateC          (original paths unchanged)
//   stateB   → stateA_1                         (was → stateA)
//   stateC   → stateA_1                         (was → stateA)
//   stateA_1 → select → stateB_1, stateC_1
//   stateB_1 → @reject                          (stateA at hs:2 is OOB)
//   stateC_1 → @reject

!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_2xhdr = !p4hir.array<2x!hdr>
!hs2 = !p4hir.header_stack<2x!hdr>
!b1 = !p4hir.bit<1>
!b32 = !p4hir.bit<32>
#int0_b1 = #p4hir.int<0> : !b1

// CHECK-LABEL: @test_branching_with_stack
module @test_branching_with_stack {
    // CHECK: p4hir.parser @BranchingWithStack
    p4hir.parser @BranchingWithStack(%cond: !b1)() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition_select
        // CHECK:   } to @stateB
        // CHECK:   } to @stateC
        p4hir.state @stateA {
            %stack = p4hir.variable ["stack"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhdr>, !b32
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @stateB
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @stateC
            }
        }
        // CHECK: p4hir.state @stateB {
        // CHECK:   p4hir.transition to @stateA_1
        p4hir.state @stateB {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateC {
        // CHECK:   p4hir.transition to @stateA_1
        p4hir.state @stateC {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition_select
        // CHECK:   } to @stateB_1
        // CHECK:   } to @stateC_1
        // CHECK: p4hir.state @stateB_1 {
        // CHECK:   p4hir.transition to @stateOutOfBound
        // CHECK: p4hir.state @stateC_1 {
        // CHECK:   p4hir.transition to @stateOutOfBound
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }
}
