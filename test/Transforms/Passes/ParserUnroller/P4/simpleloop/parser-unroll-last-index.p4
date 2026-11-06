// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s
// hdr.big.last is emitted as array_get data[nextIndex - 1]. Nothing advances
// nextIndex at runtime once .next accesses are folded, so .last must be folded
// to the same constant as the preceding .next in every clone.
// CHECK-LABEL: p4hir.state @parse_loop {
// CHECK:   p4hir.array_element_ref
// CHECK:   %[[L0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_get {{.*}}[%[[L0]]]
// CHECK-LABEL: p4hir.state @parse_loop_1 {
// CHECK-NOT:   p4hir.array_get {{.*}}[%sub{{.*}}]
// CHECK-LABEL: p4hir.state @parse_loop_2 {
// CHECK:   p4hir.array_element_ref
// CHECK:   %[[L2:.*]] = p4hir.const #int2_b32i
// CHECK:   p4hir.array_get {{.*}}[%[[L2]]]
// CHECK-LABEL: p4hir.state @parse_loop_3 {
// CHECK:   p4hir.array_element_ref
// CHECK:   %[[L3:.*]] = p4hir.const #int3_b32i
// CHECK:   p4hir.array_get {{.*}}[%[[L3]]]

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header big_t { bit<8> val; }

struct headers {
    big_t[4] big;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        transition parse_loop;
    }

    state parse_loop {
        packet.extract(hdr.big.next);
        transition select(hdr.big.last.val) {
            0xff: accept;
            default: parse_loop;
        }
    }
}
