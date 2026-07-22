// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!arr_4xhdr = !p4hir.array<4x!hdr>
!hs4 = !p4hir.header_stack<4x!hdr>

// CHECK-LABEL: @test_header_stack_loop
module @test_header_stack_loop {
    // CHECK: p4hir.parser @HSParser
    p4hir.parser @HSParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_loop
        }
        // CHECK: p4hir.state @parse_loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @parse_loop_1
        // CHECK: p4hir.state @parse_loop_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @parse_loop_2
        // CHECK: p4hir.state @parse_loop_2 {
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK:   p4hir.transition to @parse_loop_3
        // CHECK: p4hir.state @parse_loop_3 {
        // CHECK:   %[[C3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C3]]]
        // CHECK:   p4hir.transition to @reject
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


!arr_8xhdr = !p4hir.array<8x!hdr>
!hs8 = !p4hir.header_stack<8x!hdr>

// CHECK-LABEL: @test_header_stack_depth8
module @test_header_stack_depth8 {
    // CHECK: p4hir.parser @BigStackParser
    p4hir.parser @BigStackParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loop
        }
        // CHECK: p4hir.state @loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @loop_1
        // CHECK: p4hir.state @loop_1
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK: p4hir.state @loop_2
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK: p4hir.state @loop_3
        // CHECK:   %[[C3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C3]]]
        // CHECK: p4hir.state @loop_4
        // CHECK:   %[[C4:.*]] = p4hir.const #int4_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C4]]]
        // CHECK: p4hir.state @loop_5
        // CHECK:   %[[C5:.*]] = p4hir.const #int5_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C5]]]
        // CHECK: p4hir.state @loop_6
        // CHECK:   %[[C6:.*]] = p4hir.const #int6_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C6]]]
        // CHECK: p4hir.state @loop_7 {
        // CHECK:   %[[C7:.*]] = p4hir.const #int7_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C7]]]
        // CHECK:   p4hir.transition to @reject
        p4hir.state @loop {
            %stack = p4hir.variable ["stack"] : <!hs8>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs8>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs8>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_8xhdr>, !b32i
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

// 3 clones of each SCC state → chain: C→A_1→B_1→C_1→A_2→B_2→C_2→A_3→B_3→reject

// CHECK-LABEL: @test_hs_multistate
module @test_hs_multistate {
    // CHECK: p4hir.parser @HSMultiState
    p4hir.parser @HSMultiState()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @stateA
        }
        // CHECK: p4hir.state @stateA {
        // CHECK:   p4hir.transition to @stateB
        p4hir.state @stateA {
            p4hir.transition to @stateB
        }
        // CHECK: p4hir.state @stateB {
        // CHECK:   %[[B0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B0]]]
        // CHECK:   p4hir.transition to @stateC
        p4hir.state @stateB {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %stack["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32i>
            %data_ref = p4hir.struct_field_ref %stack["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xhdr>, !b32i
            p4hir.transition to @stateC
        }
        // CHECK: p4hir.state @stateC {
        // CHECK:   p4hir.transition to @stateA_1
        // CHECK: p4hir.state @stateA_1 {
        // CHECK:   p4hir.transition to @stateB_1
        // CHECK: p4hir.state @stateB_1 {
        // CHECK:   %[[B1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B1]]]
        // CHECK:   p4hir.transition to @stateC_1
        // CHECK: p4hir.state @stateC_1 {
        // CHECK:   p4hir.transition to @stateA_2
        // CHECK: p4hir.state @stateA_2 {
        // CHECK:   p4hir.transition to @stateB_2
        // CHECK: p4hir.state @stateB_2 {
        // CHECK:   %[[B2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B2]]]
        // CHECK:   p4hir.transition to @stateC_2
        // CHECK: p4hir.state @stateC_2 {
        // CHECK:   p4hir.transition to @stateA_3
        // CHECK: p4hir.state @stateA_3 {
        // CHECK:   p4hir.transition to @stateB_3
        // CHECK: p4hir.state @stateB_3 {
        // CHECK:   %[[B3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B3]]]
        // CHECK:   p4hir.transition to @reject
        p4hir.state @stateC {
            p4hir.transition to @stateA
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


// CHECK-LABEL: @test_hs_min_depth
module @test_hs_min_depth {
    // CHECK: p4hir.parser @MinDepthParser
    p4hir.parser @MinDepthParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loop
        }
        // CHECK: p4hir.state @loop {
        // CHECK:   %[[A0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[A0]]]
        // CHECK:   %[[B0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B0]]]
        // CHECK:   p4hir.transition to @loop_1
        // CHECK: p4hir.state @loop_1
        // CHECK:   %[[A1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[A1]]]
        // CHECK:   %[[B1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B1]]]
        // CHECK: p4hir.state @loop_2
        // CHECK:   %[[A2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[A2]]]
        // CHECK:   %[[B2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B2]]]
        // CHECK: p4hir.state @loop_3 {
        // CHECK:   %[[A3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[A3]]]
        // CHECK:   %[[B3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[B3]]]
        // CHECK:   p4hir.transition to @reject
        p4hir.state @loop {
            %s4 = p4hir.variable ["s4"] : <!hs4>
            %ni4_ref = p4hir.struct_field_ref %s4["nextIndex"] : <!hs4>
            %ni4 = p4hir.read %ni4_ref : <!b32i>
            %d4_ref = p4hir.struct_field_ref %s4["data"] : <!hs4>
            %e4_ref = p4hir.array_element_ref %d4_ref[%ni4] : !p4hir.ref<!arr_4xhdr>, !b32i
            %s8 = p4hir.variable ["s8"] : <!hs8>
            %ni8_ref = p4hir.struct_field_ref %s8["nextIndex"] : <!hs8>
            %ni8 = p4hir.read %ni8_ref : <!b32i>
            %d8_ref = p4hir.struct_field_ref %s8["data"] : <!hs8>
            %e8_ref = p4hir.array_element_ref %d8_ref[%ni8] : !p4hir.ref<!arr_8xhdr>, !b32i
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


!b8i = !p4hir.bit<8>
!small_h = !p4hir.header<"small_h", val: !b8i, __valid: !validity_bit>
!hs2 = !p4hir.header_stack<2x!small_h>

// CHECK-LABEL: @test_hs_fixed_index_ignored
module @test_hs_fixed_index_ignored {
    // CHECK: p4hir.parser @FixedIndexParser
    p4hir.parser @FixedIndexParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @loop
        }
        // CHECK: p4hir.state @loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @loop_1
        // CHECK: p4hir.state @loop_1
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK: p4hir.state @loop_2
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK: p4hir.state @loop_3 {
        // CHECK:   %[[C3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C3]]]
        // CHECK:   p4hir.transition to @reject
        p4hir.state @loop {
            %big = p4hir.variable ["big"] : <!hs4>
            %bigNext_ref = p4hir.struct_field_ref %big["nextIndex"] : <!hs4>
            %bigNext = p4hir.read %bigNext_ref : <!b32i>
            %bigData_ref = p4hir.struct_field_ref %big["data"] : <!hs4>
            %bigElt_ref = p4hir.array_element_ref %bigData_ref[%bigNext] : !p4hir.ref<!arr_4xhdr>, !b32i
            %small = p4hir.variable ["small"] : <!hs2>
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
