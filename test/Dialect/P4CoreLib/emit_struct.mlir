// RUN: p4mlir-opt %s --p4hir-expand-emit | FileCheck %s

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
!hs_2ht = !p4hir.header_stack<2x!header_two>
!arr_int = !p4hir.array<2 x !header_two>
!hu = !p4hir.header_union<"hu", h1: !header_one, h2: !header_two>
!struct_int = !p4hir.struct<"headers_int", t5: !hu, t6: !arr_int>
!headers_t = !p4hir.struct<"headers_t", t1: !header_one, t2: !header_two, t3: !struct_int, t4: !hs_2ht>
module {
  // CHECK-LABEL: emitTest
  p4hir.control @emitTest(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "packet"}, %arg1: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      // CHECK: %[[C1:.*]] = p4hir.const #int1_b32i
      // CHECK: %[[C0:.*]] = p4hir.const #int0_b32i
      // CHECK: %[[T1:.*]] = p4hir.struct_extract %arg1["t1"] : !headers_t
      // CHECK: p4corelib.emit %[[T1]] : !header_one to %arg0 : !p4corelib.packet_out
      // CHECK: %[[T2:.*]] = p4hir.struct_extract %arg1["t2"] : !headers_t
      // CHECK: p4corelib.emit %[[T2]] : !header_two to %arg0 : !p4corelib.packet_out
      // CHECK: %[[T3:.*]] = p4hir.struct_extract %arg1["t3"] : !headers_t
      // CHECK: %[[T5:.*]] = p4hir.struct_extract %[[T3]]["t5"] : !headers_int
      // CHECK: %[[H1:.*]] = p4hir.struct_extract %[[T5]]["h1"] : !hu
      // CHECK: p4corelib.emit %[[H1]] : !header_one to %arg0 : !p4corelib.packet_out
      // CHECK: %[[H2:.*]] = p4hir.struct_extract %[[T5]]["h2"] : !hu
      // CHECK: p4corelib.emit %[[H2]] : !header_two to %arg0 : !p4corelib.packet_out
      // CHECK: %[[T6:.*]] = p4hir.struct_extract %[[T3]]["t6"] : !headers_int
      // CHECK: %[[AR0:.*]] = p4hir.array_get %[[T6]][%[[C0]]] : !arr_2xheader_two, !b32i
      // CHECK: p4corelib.emit %[[AR0]] : !header_two to %arg0 : !p4corelib.packet_out
      // CHECK: %[[AR1:.*]] = p4hir.array_get %[[T6]][%[[C1]]] : !arr_2xheader_two, !b32i
      // CHECK: p4corelib.emit %[[AR1]] : !header_two to %arg0 : !p4corelib.packet_out
      // CHECK: %[[T4:.*]] = p4hir.struct_extract %arg1["t4"] : !headers_t
      // CHECK: %[[HSD:.*]] = p4hir.struct_extract %[[T4]]["data"] : !hs_2xheader_two
      // CHECK: %[[AR2:.*]] = p4hir.array_get %[[HSD]][%[[C0]]] : !arr_2xheader_two, !b32i
      // CHECK: p4corelib.emit %[[AR2]] : !header_two to %arg0 : !p4corelib.packet_out
      // CHECK: %[[AR3:.*]] = p4hir.array_get %[[HSD]][%[[C1]]] : !arr_2xheader_two, !b32i
      // CHECK: p4corelib.emit %[[AR3]] : !header_two to %arg0 : !p4corelib.packet_out
      p4corelib.emit %arg1 : !headers_t to %arg0 : !p4corelib.packet_out
    }
  }
}

