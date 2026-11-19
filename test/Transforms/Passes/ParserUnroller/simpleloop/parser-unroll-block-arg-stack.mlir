// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!hs4 = !p4hir.header_stack<4x!hdr>
!hdrs = !p4hir.struct<"hdrs", stack: !hs4>
!b32 = !p4hir.bit<32>

// CHECK-LABEL: @test_block_arg_stack
module @test_block_arg_stack {
    // CHECK: p4hir.parser @ArgStackParser
    p4hir.parser @ArgStackParser(%hdrs: !p4hir.ref<!hdrs>)() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loop
        }
        // CHECK: p4hir.state @loop {
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK:   p4hir.transition to @loop_1
        // CHECK: p4hir.state @loop_1
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK: p4hir.state @loop_2
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK: p4hir.state @loop_3 {
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK:   p4hir.transition to @start_outOfBound_0
        p4hir.state @loop {
            %stack_ref = p4hir.struct_field_ref %hdrs["stack"] : <!hdrs>
            p4hir.header_stack_extract_next %stack_ref : !p4hir.ref<!hs4>
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
