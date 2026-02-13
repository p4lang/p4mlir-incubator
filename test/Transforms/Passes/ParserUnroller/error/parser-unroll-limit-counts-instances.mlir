// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll="symbolic-execution-limit=7" | FileCheck %s

!b1 = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int0_b1 = #p4hir.int<0> : !b1

// The parser has 7 distinct state instances: @start, @a, @b and 4 copies of
// @loop. @loop is reached from both @a and @b, so the worklist pops it twice
// at index 0. The limit is documented as state instances, so a limit of 7
// must still unroll.

// CHECK-LABEL: p4hir.state @loop {
// CHECK:   p4hir.transition to @loop_1
// CHECK-LABEL: p4hir.state @loop_3 {
module @test_limit_counts_instances {
    p4hir.parser @LimitCountsInstances(%cond: !b1)() {
        p4hir.state @start {
            p4hir.transition_select %cond : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @a
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @b
            }
        }
        p4hir.state @a {
            p4hir.transition to @loop
        }
        p4hir.state @b {
            p4hir.transition to @loop
        }
        p4hir.state @loop {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
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
