// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test10: a runtime (non-.next) stack index must be left as-is
// — no folding, no clones (p4c keeps hdr.hs[meta.hs_next_index] too).

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK:   p4hir.array_element_ref
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @start_1

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header test_h { bit<8> field; }
struct headers { test_h[4] hs; }
struct metadata_t { bit<2> hs_next_index; }

parser p(packet_in pkt, out headers hdr, inout metadata_t meta) {
    state start {
        hdr.hs[meta.hs_next_index].setValid();
        transition accept;
    }
}
