// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @parse_ethernet
// CHECK: p4hir.state @parse_srcRouting {
// CHECK:   %[[N0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N0]]]
// CHECK:   } to @parse_ipv4
// CHECK:   } to @cont_a
// CHECK:   } to @cont_b
// CHECK: p4hir.state @cont_a {
// CHECK:   p4hir.transition to @parse_srcRouting_1
// CHECK: p4hir.state @cont_b {
// CHECK:   p4hir.transition to @parse_srcRouting_1
// CHECK: p4hir.state @parse_srcRouting_1 {
// CHECK:   %[[N1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N1]]]
// CHECK:   } to @parse_ipv4_1
// CHECK:   } to @cont_a_1
// CHECK:   } to @cont_b_1
// CHECK: p4hir.state @cont_a_1 {
// CHECK:   p4hir.transition to @parse_srcRouting_2
// CHECK: p4hir.state @cont_b_1 {
// CHECK:   p4hir.transition to @parse_srcRouting_2
// CHECK: p4hir.state @parse_srcRouting_2 {
// CHECK:   %[[N2:.*]] = p4hir.const #int2_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[N2]]]
// CHECK:   } to @parse_ipv4_2
// CHECK:   } to @cont_a_2
// CHECK:   } to @cont_b_2
// CHECK: p4hir.state @cont_a_2 {
// CHECK:   p4hir.transition to @start_outOfBound_0
// CHECK: p4hir.state @cont_b_2 {
// CHECK:   p4hir.transition to @start_outOfBound_0
// CHECK-NOT: @parse_srcRouting_3
// CHECK-NOT: @cont_a_3

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

const bit<16> TYPE_SRCROUTING = 0x1234;
const bit<16> MAX_HOPS = 3;

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
}

parser p(packet_in packet, out headers hdr) {
    int<32> index;

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        index = 0;
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_SRCROUTING: parse_srcRouting;
            default: accept;
        }
    }

    state parse_srcRouting {
        packet.extract(hdr.srcRoutes[index]);
        index = (int<32>)((int)index + 1);
        transition select(hdr.srcRoutes[index - 1].bos) {
            1: parse_ipv4;
            2: cont_a;
            default: cont_b;
        }
    }

    state cont_a {
        transition parse_srcRouting;
    }

    state cont_b {
        transition parse_srcRouting;
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }
}
