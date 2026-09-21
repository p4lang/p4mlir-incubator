// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// Simple counter-based loop that should unroll 3 times.
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @parse_loop {
// CHECK:   } to @accept
// CHECK:   } to @parse_loop_1
// CHECK: p4hir.state @parse_loop_1 {
// CHECK:   } to @accept
// CHECK:   } to @parse_loop_2
// CHECK: p4hir.state @parse_loop_2 {
// CHECK:   } to @accept
// CHECK:   } to @start_outOfBound_0
// CHECK-NOT: @parse_loop_3

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header data_t { bit<8> val; }

struct headers {
    data_t data;
}

parser p(packet_in packet, out headers hdr) {
    bit<8> counter;

    state start {
        counter = 0;
        transition parse_loop;
    }

    state parse_loop {
        counter = counter + 1;
        transition select(counter) {
            3: accept;
            default: parse_loop;
        }
    }
}
