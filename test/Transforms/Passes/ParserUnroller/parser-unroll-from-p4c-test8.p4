// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test8: a constant stack index hdr.h[0] is not a .next access
// and is kept verbatim — no folding beyond the literal, no clones.

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @parse_hdrs {
// CHECK:   p4hir.array_element_ref
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @parse_hdrs_1

@__corelib
extern packet_in { void extract<T>(out T hdr); }

header h_index1 { bit<8> index; }
header h_index2 { bit<8> index; }
header_union h_stack { h_index1 i1; h_index2 i2; }
header h_index { bit<8> index; }

struct headers { h_stack[2] h; h_index i; }

parser p(packet_in pkt, out headers hdr) {
    state start {
        transition parse_hdrs;
    }
    state parse_hdrs {
        pkt.extract(hdr.h[0].i1);
        pkt.extract(hdr.i);
        hdr.i.index = hdr.h[0].i1.index;
        transition accept;
    }
}
