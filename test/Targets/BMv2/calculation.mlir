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
  bmv2ir.header_instance @verifyChecksum0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
  bmv2ir.header_instance @computeChecksum0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
  bmv2ir.calculation @calculation_0 {
    %0 = bmv2ir.field @verifyChecksum0_ipv4["version"] -> !b4i
    %1 = bmv2ir.field @verifyChecksum0_ipv4["ihl"] -> !b4i
    %2 = bmv2ir.field @verifyChecksum0_ipv4["diffserv"] -> !b8i
    %3 = bmv2ir.field @verifyChecksum0_ipv4["totalLen"] -> !b16i
    %4 = bmv2ir.field @verifyChecksum0_ipv4["identification"] -> !b16i
    %5 = bmv2ir.field @verifyChecksum0_ipv4["flags"] -> !b3i
    %6 = bmv2ir.field @verifyChecksum0_ipv4["fragOffset"] -> !b13i
    %7 = bmv2ir.field @verifyChecksum0_ipv4["ttl"] -> !b8i
    %8 = bmv2ir.field @verifyChecksum0_ipv4["protocol"] -> !b8i
    %9 = bmv2ir.field @verifyChecksum0_ipv4["srcAddr"] -> !b32i
    %10 = bmv2ir.field @verifyChecksum0_ipv4["dstAddr"] -> !b32i
    bmv2ir.yield %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : !b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i
  } {algo = "csum16"}
  bmv2ir.calculation @calculation_1 {
    %0 = bmv2ir.field @computeChecksum0_ipv4["version"] -> !b4i
    %1 = bmv2ir.field @computeChecksum0_ipv4["ihl"] -> !b4i
    %2 = bmv2ir.field @computeChecksum0_ipv4["diffserv"] -> !b8i
    %3 = bmv2ir.field @computeChecksum0_ipv4["totalLen"] -> !b16i
    %4 = bmv2ir.field @computeChecksum0_ipv4["identification"] -> !b16i
    %5 = bmv2ir.field @computeChecksum0_ipv4["flags"] -> !b3i
    %6 = bmv2ir.field @computeChecksum0_ipv4["fragOffset"] -> !b13i
    %7 = bmv2ir.field @computeChecksum0_ipv4["ttl"] -> !b8i
    %8 = bmv2ir.field @computeChecksum0_ipv4["protocol"] -> !b8i
    %9 = bmv2ir.field @computeChecksum0_ipv4["srcAddr"] -> !b32i
    %10 = bmv2ir.field @computeChecksum0_ipv4["dstAddr"] -> !b32i
    bmv2ir.yield %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : !b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i
  } {algo = "csum16"}
}

// CHECK:  "calculations": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "algo": "csum16",
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "input": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "version"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "ihl"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "diffserv"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "totalLen"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "identification"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "flags"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "fragOffset"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "ttl"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "protocol"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "srcAddr"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "verifyChecksum0_ipv4",
// CHECK-NEXT:            "dstAddr"
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "name": "calculation_0"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "algo": "csum16",
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "input": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "version"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "ihl"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "diffserv"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "totalLen"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "identification"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "flags"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "fragOffset"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "ttl"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "protocol"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "srcAddr"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "type": "field",
// CHECK-NEXT:          "value": [
// CHECK-NEXT:            "computeChecksum0_ipv4",
// CHECK-NEXT:            "dstAddr"
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "name": "calculation_1"
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
