// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

//   A → B → A   (cycle 1, B advances hs1; size 2)
//   A → C → A   (cycle 2, C advances NO stack)
//   A → accept

!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_2xhdr = !p4hir.array<2x!hdr>
!hs2 = !p4hir.header_stack<2x!hdr>
!b1 = !p4hir.bit<1>
!b2 = !p4hir.bit<2>
!b32 = !p4hir.bit<32>
#int0_b2 = #p4hir.int<0> : !b2
#int1_b2 = #p4hir.int<1> : !b2

// CHECK-LABEL: @test_disjoint_cycles
module @test_disjoint_cycles {
    p4hir.parser @DisjointCycles(%cond: !b2)() {
        p4hir.state @start {
            p4hir.transition to @A
        }
        // CHECK: p4hir.state @A {
        // CHECK:   p4hir.transition_select
        p4hir.state @A {
            p4hir.transition_select %cond : !b2 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b2
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b2>
                    p4hir.yield %s0 : !p4hir.set<!b2>
                } to @B
                p4hir.select_case {
                    %c1 = p4hir.const #int1_b2
                    %s1 = p4hir.set (%c1) : !p4hir.set<!b2>
                    p4hir.yield %s1 : !p4hir.set<!b2>
                } to @C
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @accept
            }
        }
        p4hir.state @B {
            %stack = p4hir.variable ["hs1"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhdr>, !b32
            p4hir.transition to @A
        }
        p4hir.state @C {
            p4hir.transition to @A
        }
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }



    // CHECK-LABEL: p4hir.state @A_1
    // CHECK-LABEL: p4hir.state @B_1
    // CHECK:        %[[B1:.*]] = p4hir.const #int1_b32i
    // CHECK:        p4hir.array_element_ref {{.*}}[%[[B1]]]
    // CHECK-LABEL: p4hir.state @C_1
    // CHECK-LABEL: p4hir.state @A_2 {
    // CHECK:        } to @stateOutOfBound
    // CHECK:        } to @C_2
    // CHECK:        } to @accept
    // CHECK-LABEL: p4hir.state @C_2

    // CHECK-NOT: p4hir.state @B_2
}
