// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s
// Stack indexed by a parser variable i = 0, 1, 2, ... instead of .next: both
// hs[i] and hs[i - 1] fold to the iteration number, and the 4th loop-back is
// out of bounds.
// CHECK-LABEL: p4hir.state @parse_loop {
// CHECK:   %[[E0:.*]] = p4hir.const #int0_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E0]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G0:.*]] = p4hir.const #int0_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G0]]]
// CHECK-LABEL: p4hir.state @parse_loop_1 {
// CHECK:   %[[E1:.*]] = p4hir.const #int1_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E1]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G1:.*]] = p4hir.const #int1_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G1]]]
// CHECK-LABEL: p4hir.state @parse_loop_2 {
// CHECK:   %[[E2:.*]] = p4hir.const #int2_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E2]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G2:.*]] = p4hir.const #int2_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G2]]]
// CHECK-LABEL: p4hir.state @parse_loop_3 {
// CHECK:   %[[E3:.*]] = p4hir.const #int3_b32i
// CHECK-NEXT:   p4hir.array_element_ref {{.*}}[%[[E3]]]
// CHECK:   p4hir.binop(sub
// CHECK:   %[[G3:.*]] = p4hir.const #int3_b32i
// CHECK-NEXT:   p4hir.array_get {{.*}}[%[[G3]]]
// CHECK:   } to @accept
// CHECK:   } to @start_outOfBound_0
// CHECK-NOT: @parse_loop_4

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
        i = 0;
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
