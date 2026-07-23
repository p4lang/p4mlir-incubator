// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s

//     start → A
//     A → {B, accept}
//     B → C
//     C → {B, A}

!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_2xhdr = !p4hir.array<2x!hdr>
!hs2 = !p4hir.header_stack<2x!hdr>
!b2 = !p4hir.bit<2>
!b32 = !p4hir.bit<32>
#int0_b2 = #p4hir.int<0> : !b2

// CHECK-LABEL: @test_irreducible
module @test_irreducible {
    p4hir.parser @Irreducible(%cond: !b2)() {
        p4hir.state @start {
            p4hir.transition to @A
        }
        p4hir.state @A {
            p4hir.transition_select %cond : !b2 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b2
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b2>
                    p4hir.yield %s0 : !p4hir.set<!b2>
                } to @B
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @accept
            }
        }
        // expected-warning@below {{parser loop at state 'B' overlaps with a nested loop; outer loop will not be unrolled}}
        p4hir.state @B {
            %stack = p4hir.variable ["hs1"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhdr>, !b32
            p4hir.transition to @C
        }
        p4hir.state @C {
            p4hir.transition_select %cond : !b2 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b2
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b2>
                    p4hir.yield %s0 : !p4hir.set<!b2>
                } to @B
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @A
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

    // CHECK-LABEL: p4hir.state @A {
    // CHECK:         } to @B
    // CHECK:         } to @accept
    // CHECK-LABEL: p4hir.state @B {
    // CHECK:         %[[I0:.*]] = p4hir.const #int0_b32i
    // CHECK:         p4hir.array_element_ref {{.*}}[%[[I0]]]
    // CHECK:         p4hir.transition to @C
    // CHECK-LABEL: p4hir.state @C {
    // CHECK:         } to @B_1
    // CHECK:         } to @A_1
    // CHECK-LABEL: p4hir.state @A_1 {
    // CHECK:         } to @B_1
    // CHECK:         } to @accept
    // CHECK-LABEL: p4hir.state @B_1 {
    // CHECK:         %[[I1:.*]] = p4hir.const #int1_b32i
    // CHECK:         p4hir.array_element_ref {{.*}}[%[[I1]]]
    // CHECK:         p4hir.transition to @C_1
    // CHECK-LABEL: p4hir.state @C_1 {
    // CHECK:         } to @reject
    // CHECK:         } to @A_2
    // CHECK-LABEL: p4hir.state @A_2 {
    // CHECK:         } to @reject
    // CHECK:         } to @accept

    // CHECK-NOT: p4hir.state @B_2
    // CHECK-NOT: p4hir.state @A_3
    // CHECK-NOT: p4hir.state @C_2
}
