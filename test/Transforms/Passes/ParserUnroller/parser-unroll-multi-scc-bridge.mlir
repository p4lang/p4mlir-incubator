// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

//   start → A   (SCC_A, advances hs)
//             ↘
//              bridge (NOT in any SCC)
//             ↗
//   B → ... (SCC_B, advances hs)

!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
!b1 = !p4hir.bit<1>
!b32 = !p4hir.bit<32>
#int0_b1 = #p4hir.int<0> : !b1


// CHECK-LABEL: p4hir.state @A {
// CHECK:        %[[A0:.*]] = p4hir.const #int0_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[A0]]]
// CHECK:        } to @bridge
// CHECK:        } to @A_1

// CHECK-LABEL: p4hir.state @A_1 {
// CHECK:        %[[A1:.*]] = p4hir.const #int1_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[A1]]]
// CHECK:        } to @bridge_1
// CHECK:        } to @A_2

// CHECK-LABEL: p4hir.state @A_2 {
// CHECK:        %[[A2:.*]] = p4hir.const #int2_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[A2]]]
// CHECK:        } to @bridge_2
// CHECK:        } to @A_3

// CHECK-LABEL: p4hir.state @A_3 {
// CHECK:        %[[A3:.*]] = p4hir.const #int3_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[A3]]]
// CHECK:        } to @bridge_3
// CHECK:        } to @stateOutOfBound

// CHECK-LABEL: p4hir.state @bridge {
// CHECK:        p4hir.transition to @B

// CHECK-LABEL: p4hir.state @bridge_1 {
// CHECK:        p4hir.transition to @B_1

// CHECK-LABEL: p4hir.state @bridge_2 {
// CHECK:        p4hir.transition to @B_2

// CHECK-LABEL: p4hir.state @bridge_3 {
// CHECK:        p4hir.transition to @stateOutOfBound

// CHECK-LABEL: p4hir.state @B {
// CHECK:        %[[B0:.*]] = p4hir.const #int1_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[B0]]]
// CHECK:        } to @accept
// CHECK:        } to @B_1

// CHECK-LABEL: p4hir.state @B_1 {
// CHECK:        %[[B1:.*]] = p4hir.const #int2_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[B1]]]
// CHECK:        } to @accept
// CHECK:        } to @B_2

// CHECK-LABEL: p4hir.state @B_2 {
// CHECK:        %[[B2:.*]] = p4hir.const #int3_b32i
// CHECK:        p4hir.array_element_ref {{.*}}[%[[B2]]]
// CHECK:        } to @accept
// CHECK:        } to @stateOutOfBound

module @test_multi_scc_bridge {
    p4hir.parser @MultiSccBridge(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition to @A
        }
        p4hir.state @A {
            %stack = p4hir.variable ["hs"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @bridge
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @A
            }
        }
        p4hir.state @bridge {
            p4hir.transition to @B
        }
        p4hir.state @B {
            %stack = p4hir.variable ["hs"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @B
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
