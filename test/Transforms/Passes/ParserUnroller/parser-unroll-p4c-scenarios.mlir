// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s

!b1 = !p4hir.bit<1>
!b2 = !p4hir.bit<2>
!b32 = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit

!srcRoute = !p4hir.header<"srcRoute_t", __valid: !validity_bit>
!arr_3xsrcRoute = !p4hir.array<3x!srcRoute>
!arr_4xsrcRoute = !p4hir.array<4x!srcRoute>
!hs3 = !p4hir.header_stack<3x!srcRoute>   // MAX_HOPS = 3  (test1 / test2)
!hs4 = !p4hir.header_stack<4x!srcRoute>   // MAX_HOPS = 4  (test3)

!hstk = !p4hir.header<"h_stack", __valid: !validity_bit>
!arr_2xhstk = !p4hir.array<2x!hstk>
!hs2 = !p4hir.header_stack<2x!hstk>       // stack size 2  (test9)

//   Self-loop parse_srcRouting → parse_srcRouting (back-edge)
// CHECK-LABEL: @test1_self_loop_stack3
module @test1_self_loop_stack3 {
    // CHECK: p4hir.parser @SrcRoutingParser
    p4hir.parser @SrcRoutingParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_ethernet
        }
        // CHECK: p4hir.state @parse_ethernet
        p4hir.state @parse_ethernet {
            p4hir.transition to @parse_srcRouting
        }
        // CHECK: p4hir.state @parse_srcRouting {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @parse_srcRouting_1
        // CHECK: p4hir.state @parse_srcRouting_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @parse_srcRouting_2
        // CHECK: p4hir.state @parse_srcRouting_2 {
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK:   p4hir.transition to @reject
        // CHECK-NOT: @parse_srcRouting_3
        p4hir.state @parse_srcRouting {
            %srcRoutes = p4hir.variable ["srcRoutes"] : <!hs3>
            %nextIdx_ref = p4hir.struct_field_ref %srcRoutes["nextIndex"] : <!hs3>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %srcRoutes["data"] : <!hs3>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_3xsrcRoute>, !b32
            p4hir.transition to @parse_srcRouting
        }
        // CHECK: p4hir.state @parse_ipv4
        p4hir.state @parse_ipv4 {
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

//     parse_srcRouting → callMidle → callLast → parse_srcRouting (back-edge)
// CHECK-LABEL: @test3_multistate_loop_stack4
module @test3_multistate_loop_stack4 {
    // CHECK: p4hir.parser @SrcRouting4Parser
    p4hir.parser @SrcRouting4Parser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_ethernet
        }
        // CHECK: p4hir.state @parse_ethernet
        p4hir.state @parse_ethernet {
            p4hir.transition to @parse_srcRouting
        }
        // CHECK: p4hir.state @parse_srcRouting {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @callMidle
        p4hir.state @parse_srcRouting {
            %srcRoutes = p4hir.variable ["srcRoutes"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %srcRoutes["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %srcRoutes["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xsrcRoute>, !b32
            p4hir.transition to @callMidle
        }
        // CHECK: p4hir.state @callMidle {
        // CHECK:   p4hir.transition to @callLast
        p4hir.state @callMidle {
            %idx = p4hir.variable ["idx"] : <!b32>
            p4hir.transition to @callLast
        }
        // CHECK: p4hir.state @callLast {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @parse_srcRouting_1
        // CHECK: p4hir.state @parse_srcRouting_1 {
        // CHECK:   %[[C2:.*]] = p4hir.const #int2_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C2]]]
        // CHECK:   p4hir.transition to @callMidle_1
        // CHECK: p4hir.state @callMidle_1 {
        // CHECK:   p4hir.transition to @callLast_1
        // CHECK: p4hir.state @callLast_1 {
        // CHECK:   %[[C3:.*]] = p4hir.const #int3_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C3]]]
        // CHECK:   p4hir.transition to @reject
        // CHECK-NOT: @parse_srcRouting_2
        p4hir.state @callLast {
            %srcRoutes = p4hir.variable ["srcRoutes"] : <!hs4>
            %nextIdx_ref = p4hir.struct_field_ref %srcRoutes["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %srcRoutes["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_4xsrcRoute>, !b32
            p4hir.transition to @parse_srcRouting
        }
        // CHECK: p4hir.state @parse_ipv4
        p4hir.state @parse_ipv4 {
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

// CHECK-LABEL: @test9_finite_loop
module @test9_finite_loop {
    // CHECK: p4hir.parser @FiniteLoopParser
    p4hir.parser @FiniteLoopParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @finite_loop
        }
        // CHECK: p4hir.state @finite_loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @finite_loop_1
        // CHECK: p4hir.state @finite_loop_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @reject
        // CHECK-NOT: @finite_loop_2
        p4hir.state @finite_loop {
            %h = p4hir.variable ["h"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %h["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %h["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhstk>, !b32
            p4hir.transition to @finite_loop
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

// CHECK-LABEL: @test9_infinite_loop
module @test9_infinite_loop {
    // CHECK: p4hir.parser @InfiniteLoopParser
    p4hir.parser @InfiniteLoopParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @infinite_loop
        }
        // CHECK: p4hir.state @infinite_loop {
        // CHECK:   p4hir.transition to @infinite_loop
        // CHECK-NOT: @infinite_loop_1
        // expected-warning@below {{parser loop at state 'infinite_loop' has no header stack operations; cannot infer unroll depth}}
        p4hir.state @infinite_loop {
            %counter = p4hir.variable ["counter"] : <!b32>
            p4hir.transition to @infinite_loop
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

// CHECK-LABEL: @test9_mixed_loops
module @test9_mixed_loops {
    // CHECK: p4hir.parser @MixedLoopParser
    p4hir.parser @MixedLoopParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @start_loops
        }
        // CHECK: p4hir.state @start_loops {
        // CHECK:   p4hir.transition to @mixed_finite_loop
        p4hir.state @start_loops {
            p4hir.transition to @mixed_finite_loop
        }
        // CHECK: p4hir.state @mixed_finite_loop {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @start_loops_1
        // CHECK: p4hir.state @start_loops_1 {
        // CHECK:   p4hir.transition to @mixed_finite_loop_1
        // CHECK: p4hir.state @mixed_finite_loop_1 {
        // CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
        // CHECK:   p4hir.transition to @start_loops_2
        // CHECK: p4hir.state @start_loops_2 {
        // CHECK:   p4hir.transition to @reject
        p4hir.state @mixed_finite_loop {
            %h = p4hir.variable ["h"] : <!hs2>
            %nextIdx_ref = p4hir.struct_field_ref %h["nextIndex"] : <!hs2>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %h["data"] : <!hs2>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!arr_2xhstk>, !b32
            p4hir.transition to @start_loops
        }
        // CHECK: p4hir.state @mixed_infinite_loop {
        // CHECK:   p4hir.transition to @start_loops
        // CHECK-NOT: @start_loops_3
        p4hir.state @mixed_infinite_loop {
            %counter = p4hir.variable ["counter"] : <!b32>
            p4hir.transition to @start_loops
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
