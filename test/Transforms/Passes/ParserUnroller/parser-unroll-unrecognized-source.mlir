// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s


!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
!arr_of_hs = !p4hir.array<2x!hs4>
!b32 = !p4hir.bit<32>
#int0_b32 = #p4hir.int<0> : !b32

module @test_unrecognized_stack_source {
    p4hir.parser @UnrecognizedSourceParser()() {
        p4hir.state @start {
            p4hir.transition to @loop
        }
        // Stack source unrecognized → not unrolled: self-loop kept, no clone.
        // CHECK: p4hir.state @loop {
        // CHECK:   p4hir.transition to @loop
        // CHECK-NOT: @loop_1
        // expected-warning@below {{cannot determine identity of header stack accessed in state 'loop'; not unrolling this loop}}
        p4hir.state @loop {
            %arr = p4hir.variable ["arr"] : <!arr_of_hs>
            %c0 = p4hir.const #int0_b32
            %hs_ref = p4hir.array_element_ref %arr[%c0] : !p4hir.ref<!arr_of_hs>, !b32
            %nextIdx_ref = p4hir.struct_field_ref %hs_ref["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %hs_ref["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32
            p4hir.transition to @loop
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
