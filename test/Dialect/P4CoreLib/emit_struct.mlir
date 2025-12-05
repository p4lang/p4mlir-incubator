// RUN: p4mlir-opt %s --p4hir-expand-emit | FileCheck %s

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
!struct_int = !p4hir.struct<"headers_t", t3: !header_one>
!hs_2ht = !p4hir.header_stack<2x!header_two>
!headers_t = !p4hir.struct<"headers_t", t1: !header_one, t2: !header_two, t3: !struct_int, t4: !hs_2ht>
module {
  // CHECK-LABEL: emitTest
  p4hir.control @emitTest(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "packet"}, %arg1: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      // CHECK-COUNT-5: p4corelib.emit
      p4corelib.emit %arg1 : !headers_t to %arg0 : !p4corelib.packet_out
    }
  }
}

