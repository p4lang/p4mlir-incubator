// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s

// Nothing writes nextIndex at runtime once .next accesses are folded, so
// hs.lastIndex read in @done after the loop must be folded to the index of
// the last extracted element on each exit path: 0, 1, 2 and 3.

// CHECK-LABEL: p4hir.state @done {
// CHECK-NOT:   binop(sub
// CHECK:   %[[I0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.assign %[[I0]], %idx_field_ref
// CHECK-LABEL: p4hir.state @done_1 {
// CHECK-NOT:   binop(sub
// CHECK:   %[[I1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.assign %[[I1]], %idx_field_ref
// CHECK-LABEL: p4hir.state @done_2 {
// CHECK-NOT:   binop(sub
// CHECK:   %[[I2:.*]] = p4hir.const #int2_b32i
// CHECK:   p4hir.assign %[[I2]], %idx_field_ref
// CHECK-LABEL: p4hir.state @done_3 {
// CHECK-NOT:   binop(sub
// CHECK:   %[[I3:.*]] = p4hir.const #int3_b32i
// CHECK:   p4hir.assign %[[I3]], %idx_field_ref

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header h_t { bit<8> val; }

struct headers {
    h_t[4] hs;
    bit<32> idx;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        transition loop;
    }
    state loop {
        packet.extract(hdr.hs.next);
        transition select(hdr.hs.last.val) {
            0: done;
            default: loop;
        }
    }
    state done {
        hdr.idx = hdr.hs.lastIndex;
        transition accept;
    }
}
