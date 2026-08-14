// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll="symbolic-execution-limit=1" -verify-diagnostics | FileCheck %s

!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

module @test_symbolic_execution_limit {
    // expected-warning@below {{symbolic execution exceeded 1 state instances; skipping unroll for parser 'SymExecLimitParser'}}
    p4hir.parser @SymExecLimitParser()() {
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   p4hir.transition to @parse_loop
        // CHECK-NOT: @parse_loop_1
        p4hir.state @parse_loop {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
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
