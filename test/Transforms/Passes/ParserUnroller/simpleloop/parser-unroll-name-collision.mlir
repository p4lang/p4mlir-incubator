// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

// The test has existing @parse_loop_1 and @start_outOfBound_0
// states to check that we increment the index correctly.

!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_2xhdr = !p4hir.array<2x!hdr>
!hs2 = !p4hir.header_stack<2x!hdr>

// CHECK-LABEL: @test_name_collision
module @test_name_collision {
    // CHECK: p4hir.parser @P
    p4hir.parser @P()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK:   p4hir.transition to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   p4hir.header_stack_extract_next
        // CHECK:   p4hir.transition to @start_outOfBound_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   p4hir.transition to @accept
        // CHECK: p4hir.state @start_outOfBound_0 {
        // CHECK:   p4hir.transition to @accept
        // CHECK: p4hir.state @start_outOfBound_1 {
        // CHECK:   p4hir.parser_reject
        // CHECK-NOT: @parse_loop_3
        p4hir.state @parse_loop {
            %stack = p4hir.variable ["stack"] : <!hs2>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs2>
            p4hir.transition to @parse_loop
        }
        p4hir.state @parse_loop_1 {
            p4hir.transition to @accept
        }
        p4hir.state @start_outOfBound_0 {
            p4hir.transition to @accept
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
