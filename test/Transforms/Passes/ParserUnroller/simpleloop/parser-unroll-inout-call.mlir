// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

#inout = #p4hir<dir inout>
#int0_b32i = #p4hir.int<0> : !b32i
#int5_b32i = #p4hir.int<5> : !b32i

// set_five(counter) is lowered to a copy-in, p4hir.call and copy-out. The
// call writes x_inout_arg through its inout argument, so counter is unknown
// after the copy-out and every clone keeps its exit to @x.

// CHECK-LABEL: p4hir.state @loop {
// CHECK:   } to @x
// CHECK:   } to @loop_1
// CHECK-LABEL: p4hir.state @loop_1 {
// CHECK:   } to @x
// CHECK-LABEL: p4hir.state @loop_2 {
// CHECK:   } to @x
// CHECK-LABEL: p4hir.state @loop_3 {
// CHECK:   } to @x
module @test_inout_call {
    p4hir.func @set_five(!p4hir.ref<!b32i> {p4hir.dir = #inout, p4hir.param_name = "x"})
    p4hir.parser @InoutCall()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %zero = p4hir.const #int0_b32i
            p4hir.assign %zero, %counter : <!b32i>
            p4hir.transition to @loop
        }
        p4hir.state @loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.scope {
                %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
                %copyIn = p4hir.read %counter : <!b32i>
                p4hir.assign %copyIn, %x_inout_arg : <!b32i>
                p4hir.call @test_inout_call::@set_five (%x_inout_arg) : (!p4hir.ref<!b32i>) -> ()
                %copyOut = p4hir.read %x_inout_arg : <!b32i>
                p4hir.assign %copyOut, %counter : <!b32i>
            }
            %counterVal = p4hir.read %counter : <!b32i>
            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int5_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
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
