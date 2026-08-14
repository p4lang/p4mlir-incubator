// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll 2>&1 | FileCheck %s
// CHECK: warning:{{.*}}infinite_loop{{.*}}no header stack operations
// finite_loop unrolls once (h_stack[2]) and its last iteration -> @reject.
// The counter-driven infinite_loop keeps its self-loop (no clone).  The
// start_loops SCC is cloned per M (start_loops_1, start_loops_2).
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK: p4hir.state @start_loops {
// CHECK: p4hir.state @finite_loop {
// CHECK:   %[[F0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[F0]]]
// CHECK:   } to @accept
// CHECK:   } to @finite_loop_1
// CHECK: p4hir.state @finite_loop_1 {
// CHECK:   %[[F1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[F1]]]
// CHECK:   } to @accept
// CHECK:   } to @start_outOfBound_0
// CHECK: p4hir.state @mixed_finite_loop {
// CHECK:   %[[M0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[M0]]]
// CHECK:   } to @start_loops_1
// CHECK: p4hir.state @mixed_infinite_loop {
// CHECK: p4hir.state @start_loops_1 {
// CHECK: p4hir.state @mixed_finite_loop_1 {
// CHECK:   %[[M1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[M1]]]
// CHECK:   } to @start_loops_2
// CHECK: p4hir.state @mixed_infinite_loop_1 {
// CHECK: p4hir.state @infinite_loop {
// CHECK:   p4hir.transition to @infinite_loop
// CHECK-NOT: @infinite_loop_1

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header h_stack {
    bit<8> i1;
    bit<8> i2;
}

header h_index {
    bit<8> index;
    bit<8> counter;
}

struct headers {
    h_stack[2] h;
    h_index    i;
}

parser p(packet_in pkt, out headers hdr) {
    state start {
        pkt.extract(hdr.i);
        hdr.i.counter = 0;
        transition start_loops;
    }
    state start_loops {
        hdr.i.counter = hdr.i.counter + 1;
        transition select(hdr.i.index) {
            0: mixed_finite_loop;
            1: mixed_infinite_loop;
            2: infinite_loop;
            3: finite_loop;
            default: reject;
        }
    }
    state finite_loop {
        hdr.i.counter = hdr.i.counter + 1;
        pkt.extract(hdr.h.next);
        transition select(hdr.h.last.i1) {
            2: accept;
            default: finite_loop;
        }
    }
    state mixed_finite_loop {
        pkt.extract(hdr.h.next);
        transition select(hdr.h.last.i2) {
            1: start_loops;
            2: accept;
        }
    }
    state mixed_infinite_loop {
        transition select(hdr.i.counter) {
            3: accept;
            default: start_loops;
        }
    }
    state infinite_loop {
        transition infinite_loop;
    }
}
