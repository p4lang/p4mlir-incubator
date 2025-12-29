// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s
!HashAlgorithm = !p4hir.enum<"HashAlgorithm", crc32, crc32_custom, crc16, crc16_custom, random, identity, csum16, xor16>
!b12i = !p4hir.bit<12>
!b13i = !p4hir.bit<13>
!b16i = !p4hir.bit<16>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!type_O = !p4hir.type_var<"O">
!type_T = !p4hir.type_var<"T">
#in = #p4hir<dir in>
#inout = #p4hir<dir inout>
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>
#int-1_b8i = #p4hir.int<255> : !b8i
module {
  %true = p4hir.const #true
  bmv2ir.header_instance @computeChecksum0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
  bmv2ir.header_instance @verifyChecksum0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
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
  bmv2ir.checksum @checksum_0
   target @verifyChecksum0_ipv4["hdrChecksum"]
   type "generic"
   calculation @calculation_0
   update false
   if_cond {
    bmv2ir.yield %true : !p4hir.bool
  }
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
  bmv2ir.checksum @checksum_1
   target @computeChecksum0_ipv4["hdrChecksum"]
   type "generic"
   calculation @calculation_1
   update true
   if_cond {
    bmv2ir.yield %true : !p4hir.bool
  }
}

// CHECK:  "checksums": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "calculation": "calculation_0",
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "if_cond": true,
// CHECK-NEXT:      "name": "checksum_0",
// CHECK-NEXT:      "target": [
// CHECK-NEXT:        "verifyChecksum0_ipv4",
// CHECK-NEXT:        "hdrChecksum"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "type": "generic",
// CHECK-NEXT:      "update": false,
// CHECK-NEXT:      "verify": true
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "calculation": "calculation_1",
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "if_cond": true,
// CHECK-NEXT:      "name": "checksum_1",
// CHECK-NEXT:      "target": [
// CHECK-NEXT:        "computeChecksum0_ipv4",
// CHECK-NEXT:        "hdrChecksum"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "type": "generic",
// CHECK-NEXT:      "update": true,
// CHECK-NEXT:      "verify": false
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
