// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s

// The select reads hs.lastIndex after hs.next is extracted, so iteration k
// selects on k. The exit fires at k=2: @loop_2 goes to accept and there is no
// @loop_3.

// CHECK-LABEL: p4hir.state @loop {
// CHECK:   } to @loop_1
// CHECK-LABEL: p4hir.state @loop_1 {
// CHECK:   } to @loop_2
// CHECK-LABEL: p4hir.state @loop_2 {
// CHECK:   } to @accept
// CHECK-NOT: @loop_3

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header h_t { bit<8> val; }

struct headers {
    h_t[4] hs;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        transition loop;
    }
    state loop {
        packet.extract(hdr.hs.next);
        transition select(hdr.hs.lastIndex) {
            2: accept;
            default: loop;
        }
    }
}
