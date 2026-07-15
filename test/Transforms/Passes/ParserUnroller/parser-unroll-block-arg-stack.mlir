// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s


!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!hs4 = !p4hir.header_stack<4x!hdr>
!hdrs = !p4hir.struct<"hdrs", stack: !hs4>
!b32 = !p4hir.bit<32>

// CHECK-LABEL: @test_block_arg_stack
module @test_block_arg_stack {
    // CHECK: p4hir.parser @ArgStackParser
    p4hir.parser @ArgStackParser(%hdrs: !p4hir.ref<!hdrs>)() {
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
            %stack_ref = p4hir.struct_field_ref %hdrs["stack"] : <!hdrs>
            %nextIdx_ref = p4hir.struct_field_ref %stack_ref["nextIndex"] : <!hs4>
            %nextIdx = p4hir.read %nextIdx_ref : <!b32>
            %data_ref = p4hir.struct_field_ref %stack_ref["data"] : <!hs4>
            %elt_ref = p4hir.array_element_ref %data_ref[%nextIdx] : !p4hir.ref<!p4hir.array<4x!hdr>>, !b32
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
