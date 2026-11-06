// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-wrap-extract-next -p4hir-parser-unroll | FileCheck %s
// h[3] with two .next extracts per iteration: the second iteration extracts h[2] and then
// overflows. Its clone keeps the counter update and the h[2] extract that run before the
// failing access, then goes to @start_outOfBound_0.
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @loop {
// CHECK:   p4hir.header_stack_extract_next
// CHECK:   p4hir.header_stack_extract_next
// CHECK:   } to @accept{{$}}
// CHECK:   } to @loop_1{{$}}
// CHECK: p4hir.state @loop_1 {
// CHECK:   p4hir.assign {{.*}} : <!b8i>
// CHECK:   p4hir.header_stack_extract_next
// CHECK:   p4corelib.extract_header
// CHECK-NOT: p4corelib.extract_header
// CHECK:   p4hir.transition to @start_outOfBound_0
// CHECK-NEXT: }
// CHECK: p4hir.state @start_outOfBound_0 {
// CHECK-NEXT: p4hir.parser_reject with error #error_StackOutOfBounds

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header h_stack {
    bit<8> i1;
}

header h_index {
    bit<8> counter;
}

struct headers {
    h_stack[3] h;
    h_index    i;
}

parser p(packet_in pkt, out headers hdr) {
    state start {
        pkt.extract(hdr.i);
        transition loop;
    }
    state loop {
        hdr.i.counter = hdr.i.counter + 1;
        pkt.extract(hdr.h.next);
        pkt.extract(hdr.h.next);
        transition select(hdr.h.last.i1) {
            0: accept;
            default: loop;
        }
    }
}
