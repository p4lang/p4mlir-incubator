// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s

//   start -> loopA -> loopA  (back-edge, has hs4)
//   start -> loopX -> loopX  (back-edge, no stack)

!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
!b1 = !p4hir.bit<1>

// CHECK-LABEL: @test_multi_loop
module @test_multi_loop {
    // CHECK: p4hir.parser @ParserWithStack
    p4hir.parser @ParserWithStack()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loopA
        }
        // CHECK: p4hir.state @loopA {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @loopA_1
        // CHECK: p4hir.state @loopA_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @loopA_2
        // CHECK: p4hir.state @loopA_2 {
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK:   p4hir.transition to @loopA_3
        // CHECK: p4hir.state @loopA_3 {
        // CHECK:   %[[C3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C3]]]
        // CHECK:   p4hir.transition to @start_outOfBound_0
        // CHECK-NOT: @loopA_4
        p4hir.state @loopA {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.transition to @loopA
        }
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }

    // CHECK: p4hir.parser @ParserNoStack
    p4hir.parser @ParserNoStack()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loopX
        }
        // CHECK: p4hir.state @loopX {
        // CHECK:   p4hir.transition to @loopX
        // CHECK-NOT: @loopX_1
        // expected-warning@below {{parser loop at state 'loopX' has no header stack operations and no select exit condition; cannot unroll}}
        p4hir.state @loopX {
            p4hir.transition to @loopX
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
