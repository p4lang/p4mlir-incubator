// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s
// Same as parser-unroll-explicit-index.p4, but i starts at 2: only hs[2] and
// hs[3] are in bounds, so the loop-back from @parse_loop_1 must reject with
// StackOutOfBounds instead of producing constant indices 4 and 5.
// CHECK-LABEL: p4hir.state @parse_loop {
// CHECK:   %[[E2:.*]] = p4hir.const #int2_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E2]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G2:.*]] = p4hir.const #int2_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G2]]]
// CHECK-LABEL: p4hir.state @parse_loop_1 {
// CHECK:   %[[E3:.*]] = p4hir.const #int3_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E3]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G3:.*]] = p4hir.const #int3_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G3]]]
// CHECK-NOT: @parse_loop_2
// CHECK:   } to @start_outOfBound_0
// CHECK-NOT: @parse_loop_2

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header h_t { bit<8> val; }

struct headers {
    h_t[4] hs;
}

parser p(packet_in packet, out headers hdr) {
    bit<32> i;

    state start {
        i = 2;
        transition parse_loop;
    }

    state parse_loop {
        packet.extract(hdr.hs[i]);
        i = i + 1;
        transition select(hdr.hs[i - 1].val) {
            0: accept;
            default: parse_loop;
        }
    }
}
