// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s
!b12i = !p4hir.bit<12>
!b13i = !p4hir.bit<13>
!b16i = !p4hir.bit<16>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!packet_out = !p4hir.extern<"packet_out">
!validity_bit = !p4hir.validity.bit
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
#int-1_b8i = #p4hir.int<255> : !b8i
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
module {
  bmv2ir.header_instance @deparser0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
  bmv2ir.header_instance @deparser0_ethernet : !bmv2ir.header<"ethernet_t", [dstAddr:!p4hir.bit<48>, srcAddr:!p4hir.bit<48>, etherType:!p4hir.bit<16>], max_length = 14>
  bmv2ir.deparser @deparser order [@deparser0_ethernet, @deparser0_ipv4]

}

// CHECK:  "deparsers": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "name": "deparser",
// CHECK-NEXT:      "order": [
// CHECK-NEXT:        "deparser0_ethernet",
// CHECK-NEXT:        "deparser0_ipv4"
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
