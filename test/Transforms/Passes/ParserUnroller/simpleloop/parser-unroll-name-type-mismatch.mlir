// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#int7_b8i = #p4hir.int<7> : !b8i
#int7_b32i = #p4hir.int<7> : !b32i

// @start's "t" is bit<8> and @loop's "t" is bit<32>: they share a name but
// are different variables and must not be canonicalized to one id. The
// select on @loop's "t" stays unresolved and every clone keeps its edge to @x.

// CHECK-LABEL: p4hir.state @loop {
// CHECK:   } to @x
// CHECK:   } to @loop_1
// CHECK-LABEL: p4hir.state @loop_3 {
// CHECK:   } to @x
module @test_name_type_mismatch {
    p4hir.parser @NameTypeMismatch()() {
        p4hir.state @start {
            %t = p4hir.variable ["t"] : <!b8i>
            %seven = p4hir.const #int7_b8i
            p4hir.assign %seven, %t : <!b8i>
            p4hir.transition to @loop
        }
        p4hir.state @loop {
            %t = p4hir.variable ["t"] : <!b32i>
            %stack = p4hir.variable ["stack"] : <!hs4>
            p4hir.header_stack_extract_next %stack : !p4hir.ref<!hs4>
            %tVal = p4hir.read %t : <!b32i>
            p4hir.transition_select %tVal : !b32i {
                p4hir.select_case {
                    %c7 = p4hir.const #int7_b32i
                    %s7 = p4hir.set (%c7) : !p4hir.set<!b32i>
                    p4hir.yield %s7 : !p4hir.set<!b32i>
                } to @x
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @loop
            }
        }
        p4hir.state @x {
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
