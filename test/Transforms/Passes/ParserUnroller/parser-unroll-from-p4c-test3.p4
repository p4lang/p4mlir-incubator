// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// Multi-state loop {parse_srcRouting, callMidle, callLast}, 2 extracts/iter of
// srcRoutes[4] → 1 clone of each.  callLast's loop-back to parse_srcRouting is
// rewired to the clone; the clone's callLast_1 loop-back hits @reject.
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @parse_ethernet
// CHECK: p4hir.state @parse_srcRouting {
// CHECK:   %[[N0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N0]]]
// CHECK:   } to @callMidle
// CHECK: p4hir.state @callLast {
// CHECK:   %[[N1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N1]]]
// CHECK:   } to @parse_srcRouting_1
// CHECK: p4hir.state @callMidle {
// CHECK:   p4hir.transition to @callLast
// CHECK: p4hir.state @parse_srcRouting_1 {
// CHECK:   %[[N2:.*]] = p4hir.const #int2_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N2]]]
// CHECK:   } to @callMidle_1
// CHECK: p4hir.state @callLast_1 {
// CHECK:   %[[N3:.*]] = p4hir.const #int3_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N3]]]
// CHECK:   } to @stateOutOfBound
// CHECK: p4hir.state @callMidle_1 {
// CHECK:   p4hir.transition to @callLast_1
// CHECK-NOT: @parse_srcRouting_2
// CHECK-NOT: @callMidle_2
// CHECK: p4hir.state @parse_ipv4 {

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

const bit<16> TYPE_SRCROUTING = 0x1234;
const bit<16> MAX_HOPS = 4;

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header srcRoute_t {
    bit<2>  bos;
    bit<15> port;
}

header ipv4_t { bit<8> ttl; }

struct headers {
    ethernet_t           ethernet;
    srcRoute_t[MAX_HOPS] srcRoutes;
    ipv4_t               ipv4;
    bit<32>              index;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        hdr.index = 0;
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_SRCROUTING: parse_srcRouting;
            default: accept;
        }
    }

    state parse_srcRouting {
        packet.extract(hdr.srcRoutes.next);
        transition select(hdr.srcRoutes.last.bos) {
            1: parse_ipv4;
            2: callMidle;
            default: callMidle;
        }
    }

    state callLast {
        packet.extract(hdr.srcRoutes.next);
        transition select(hdr.srcRoutes.last.bos) {
            1: parse_ipv4;
            default: parse_srcRouting;
        }
    }

    state callMidle {
        hdr.index = hdr.index + 1;
        transition callLast;
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }
}
