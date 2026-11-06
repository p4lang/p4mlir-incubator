// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

!b1 = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int0_b1 = #p4hir.int<0> : !b1
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int5_b32i = #p4hir.int<5> : !b32i

// @a sets v=1 and @b sets v=2 before both reach @c, which selects on v.
// @c must be specialised per value of v: merging both paths into one
// instance resolves the select for v=1 only and turns the v=2 edge to @y
// into a reject.

// CHECK-LABEL: p4hir.state @a {
// CHECK:   p4hir.transition to @c{{$}}
// CHECK-LABEL: p4hir.state @b {
// CHECK:   p4hir.transition to @c_1{{$}}
// CHECK-LABEL: p4hir.state @c {
// CHECK:   } to @x
// CHECK-LABEL: p4hir.state @c_1 {
// CHECK:   } to @y
module @test_select_merge {
    p4hir.parser @SelectMerge(%cond: !b1)() {
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
            %v = p4hir.variable ["v"] : <!b32i>
            %one = p4hir.const #int1_b32i
            p4hir.assign %one, %v : <!b32i>
            p4hir.transition to @c
        }
        p4hir.state @b {
            %v = p4hir.variable ["v"] : <!b32i>
            %two = p4hir.const #int2_b32i
            p4hir.assign %two, %v : <!b32i>
            p4hir.transition to @c
        }
        p4hir.state @c {
            %v = p4hir.variable ["v"] : <!b32i>
            %vVal = p4hir.read %v : <!b32i>
            p4hir.transition_select %vVal : !b32i {
                p4hir.select_case {
                    %c1 = p4hir.const #int1_b32i
                    %s1 = p4hir.set (%c1) : !p4hir.set<!b32i>
                    p4hir.yield %s1 : !p4hir.set<!b32i>
                } to @x
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @y
            }
        }
        p4hir.state @x {
            p4hir.transition to @accept
        }
        p4hir.state @y {
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
