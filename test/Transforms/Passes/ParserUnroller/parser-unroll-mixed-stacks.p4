// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// Depth from the SCC's big_stack[4], not min(4,2): 3 clones.  Every clone keeps
// its exit to @parse_small; only the loop-back advances, last one → @reject.
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @parse_loop {
// CHECK:   } to @parse_small
// CHECK:   } to @parse_loop_1
// CHECK: p4hir.state @parse_loop_1 {
// CHECK:   } to @parse_small
// CHECK:   } to @parse_loop_2
// CHECK: p4hir.state @parse_loop_2 {
// CHECK:   } to @parse_small
// CHECK:   } to @parse_loop_3
// CHECK: p4hir.state @parse_loop_3 {
// CHECK:   } to @parse_small
// CHECK:   } to @stateOutOfBound
// CHECK-NOT: @parse_loop_4
// CHECK: p4hir.state @parse_small {

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header big_t   { bit<8> val; }
header small_t { bit<8> val; }

struct headers {
    big_t[4]   big;
    small_t[2] small;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        transition parse_loop;
    }

    state parse_loop {
        packet.extract(hdr.big.next);
        transition select(hdr.big.last.val) {
            0xff: parse_small;
            default: parse_loop;
        }
    }

    state parse_small {
        packet.extract(hdr.small.next);
        transition accept;
    }
}
