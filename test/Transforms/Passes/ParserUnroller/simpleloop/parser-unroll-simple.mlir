// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


// CHECK-LABEL: @test_parser
module @test_parser {
    // CHECK: p4hir.parser @MyParser
    p4hir.parser @MyParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            // CHECK: p4hir.transition to @parse_loop
            p4hir.transition to @parse_loop
        }
        // No header stack -> not unrolled: self-loop preserved, no clone.
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.transition to @parse_loop
        // CHECK-NOT: @parse_loop_1
        p4hir.state @parse_loop {
            p4hir.transition to @parse_loop
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
