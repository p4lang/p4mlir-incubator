// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next | FileCheck %s
// Only extract(hs.next) is wrapped; plain header extracts and hs.last reads stay as they are.
// CHECK-LABEL: p4hir.state @start {
// CHECK-NOT:   p4hir.header_stack_extract_next
// CHECK:       p4corelib.extract_header
// CHECK-LABEL: p4hir.state @parse_hops {
// CHECK:       p4hir.header_stack_extract_next %{{.*}} : !p4hir.ref<!hs_3xhop_t> {
// CHECK:         p4hir.array_element_ref
// CHECK:         p4corelib.extract_header
// CHECK:         p4hir.assign
// CHECK-NEXT:  }
// CHECK-NOT:   p4hir.header_stack_extract_next
// CHECK:       p4hir.array_get
// CHECK:       p4hir.transition_select

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header eth_t {
    bit<16> etherType;
}

header hop_t {
    bit<8> port;
}

struct headers {
    eth_t    eth;
    hop_t[3] hops;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        packet.extract(hdr.eth);
        transition parse_hops;
    }

    state parse_hops {
        packet.extract(hdr.hops.next);
        transition select(hdr.hops.last.port) {
            0: accept;
            default: parse_hops;
        }
    }
}
