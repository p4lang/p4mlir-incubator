// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!b4i = !p4hir.bit<4>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>
!Meta = !p4hir.struct<"Meta", counter: !b32i>

#int0_b4i = #p4hir.int<0> : !b4i
#int1_b32i = #p4hir.int<1> : !b32i

// Each parser sets the select variable to 1 and then overwrites it with an
// unknown value. The select must stay unresolved, so the loop is unrolled
// up to the stack size instead of exiting right away to @accept.

// meta.counter = 1; meta = %m;
// CHECK-LABEL: @test_parent_assign
module @test_parent_assign {
    p4hir.parser @ParentAssignParser(%m: !Meta)() {
        p4hir.state @start {
            %meta = p4hir.variable ["meta"] : <!Meta>
            %counter_ref = p4hir.struct_field_ref %meta["counter"] : <!Meta>
            %one = p4hir.const #int1_b32i
            p4hir.assign %one, %counter_ref : <!b32i>
            p4hir.assign %m, %meta : <!Meta>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK: p4hir.state @parse_loop_3 {
        p4hir.state @parse_loop {
            %meta = p4hir.variable ["meta"] : <!Meta>
            %counter_ref = p4hir.struct_field_ref %meta["counter"] : <!Meta>
            %counterVal = p4hir.read %counter_ref : <!b32i>

            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i

            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int1_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
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

// meta = {counter = 1}; meta.counter = %c;
// CHECK-LABEL: @test_field_assign
module @test_field_assign {
    p4hir.parser @FieldAssignParser(%c: !b32i)() {
        p4hir.state @start {
            %meta = p4hir.variable ["meta"] : <!Meta>
            %init = p4hir.const #p4hir.aggregate<[#int1_b32i]> : !Meta
            p4hir.assign %init, %meta : <!Meta>
            %counter_ref = p4hir.struct_field_ref %meta["counter"] : <!Meta>
            p4hir.assign %c, %counter_ref : <!b32i>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK: p4hir.state @parse_loop_3 {
        p4hir.state @parse_loop {
            %meta = p4hir.variable ["meta"] : <!Meta>
            %metaVal = p4hir.read %meta : <!Meta>
            %counterVal = p4hir.struct_extract %metaVal["counter"] : !Meta

            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i

            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int1_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
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

// counter = 1; counter[3:0] = 0;
// CHECK-LABEL: @test_assign_slice
module @test_assign_slice {
    p4hir.parser @AssignSliceParser()() {
        p4hir.state @start {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %one = p4hir.const #int1_b32i
            p4hir.assign %one, %counter : <!b32i>
            %zero = p4hir.const #int0_b4i
            p4hir.assign_slice %zero, %counter[3 : 0] : !b4i -> <!b32i>
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK: p4hir.state @parse_loop_3 {
        p4hir.state @parse_loop {
            %counter = p4hir.variable ["counter"] : <!b32i>
            %counterVal = p4hir.read %counter : <!b32i>

            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i

            p4hir.transition_select %counterVal : !b32i {
                p4hir.select_case {
                    %exitVal = p4hir.const #int1_b32i
                    %exitSet = p4hir.set (%exitVal) : !p4hir.set<!b32i>
                    p4hir.yield %exitSet : !p4hir.set<!b32i>
                } to @accept
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_loop
            }
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
