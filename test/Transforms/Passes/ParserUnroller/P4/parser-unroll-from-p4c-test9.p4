// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll 2>&1 | FileCheck %s
// CHECK: warning:{{.*}}infinite_loop{{.*}}no header stack operations
// counter is bit<2>, so mixed_infinite_loop, which exits only at counter == 3,
// cycles through all four counter values and is unrolled per (counter, h index).
// finite_loop and mixed_finite_loop are bounded by h_stack[2] and overflow to
// @start_outOfBound_0; finite_loop_7..10 keep the counter update that runs before
// the overflowing extract. The stackless infinite_loop keeps its self-loop (no clone).
// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK:   p4hir.transition to @start_loops{{$}}
// CHECK: p4hir.state @start_loops {
// CHECK:   } to @mixed_finite_loop{{$}}
// CHECK:   } to @mixed_infinite_loop{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @finite_loop {
// CHECK:   %[[FINITELOOPE:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOPE]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_1{{$}}
// CHECK: p4hir.state @finite_loop_1 {
// CHECK:   %[[FINITELOOP1E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP1E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_7{{$}}
// CHECK: p4hir.state @finite_loop_2 {
// CHECK:   %[[FINITELOOP2E:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP2E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_3{{$}}
// CHECK: p4hir.state @finite_loop_3 {
// CHECK:   %[[FINITELOOP3E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP3E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_8{{$}}
// CHECK: p4hir.state @finite_loop_4 {
// CHECK:   %[[FINITELOOP4E:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP4E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_5{{$}}
// CHECK: p4hir.state @finite_loop_5 {
// CHECK:   %[[FINITELOOP5E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP5E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_9{{$}}
// CHECK: p4hir.state @finite_loop_6 {
// CHECK:   %[[FINITELOOP6E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[FINITELOOP6E]]]
// CHECK:   } to @accept{{$}}
// CHECK:   } to @finite_loop_10{{$}}
// CHECK: p4hir.state @mixed_finite_loop {
// CHECK:   %[[MIXEDFINITELOOPE:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOPE]]]
// CHECK:   } to @start_loops_1{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_2{{$}}
// CHECK: p4hir.state @start_loops_1 {
// CHECK:   } to @mixed_finite_loop_1{{$}}
// CHECK:   } to @mixed_infinite_loop_1{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_1{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_1 {
// CHECK:   %[[MIXEDFINITELOOP1E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP1E]]]
// CHECK:   } to @start_loops_3{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_1 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_4{{$}}
// CHECK: p4hir.state @start_loops_2 {
// CHECK:   } to @mixed_finite_loop_2{{$}}
// CHECK:   } to @mixed_infinite_loop_2{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_2{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_2 {
// CHECK:   %[[MIXEDFINITELOOP2E:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP2E]]]
// CHECK:   } to @start_loops_4{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_2 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_5{{$}}
// CHECK: p4hir.state @start_loops_3 {
// CHECK:   } to @start_outOfBound_0{{$}}
// CHECK:   } to @mixed_infinite_loop_3{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_7{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_3 {
// CHECK:   %[[MIXEDFINITELOOP3E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP3E]]]
// CHECK:   } to @start_loops_6{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_3 {
// CHECK:   } to @accept{{$}}
// CHECK: p4hir.state @start_loops_4 {
// CHECK:   } to @mixed_finite_loop_3{{$}}
// CHECK:   } to @mixed_infinite_loop_4{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_3{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_4 {
// CHECK:   %[[MIXEDFINITELOOP4E:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP4E]]]
// CHECK:   } to @start_loops_7{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_4 {
// CHECK:   } to @accept{{$}}
// CHECK: p4hir.state @start_loops_5 {
// CHECK:   } to @mixed_finite_loop_4{{$}}
// CHECK:   } to @mixed_infinite_loop_5{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_4{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_5 {
// CHECK:   %[[MIXEDFINITELOOP5E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP5E]]]
// CHECK:   } to @start_loops_8{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_5 {
// CHECK:   } to @accept{{$}}
// CHECK: p4hir.state @start_loops_6 {
// CHECK:   } to @start_outOfBound_0{{$}}
// CHECK:   } to @mixed_infinite_loop_6{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_8{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_finite_loop_6 {
// CHECK:   %[[MIXEDFINITELOOP6E:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[MIXEDFINITELOOP6E]]]
// CHECK:   } to @start_loops_10{{$}}
// CHECK:   } to @accept{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_6 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_8{{$}}
// CHECK: p4hir.state @start_loops_7 {
// CHECK:   } to @mixed_finite_loop_5{{$}}
// CHECK:   } to @mixed_infinite_loop_7{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_5{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_7 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_9{{$}}
// CHECK: p4hir.state @start_loops_8 {
// CHECK:   } to @start_outOfBound_0{{$}}
// CHECK:   } to @mixed_infinite_loop_8{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_9{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_8 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_10{{$}}
// CHECK: p4hir.state @start_loops_9 {
// CHECK:   } to @mixed_finite_loop_6{{$}}
// CHECK:   } to @mixed_infinite_loop_9{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_6{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_9 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_1{{$}}
// CHECK: p4hir.state @start_loops_10 {
// CHECK:   } to @start_outOfBound_0{{$}}
// CHECK:   } to @mixed_infinite_loop_10{{$}}
// CHECK:   } to @infinite_loop{{$}}
// CHECK:   } to @finite_loop_10{{$}}
// CHECK:   } to @reject{{$}}
// CHECK: p4hir.state @mixed_infinite_loop_10 {
// CHECK:   } to @accept{{$}}
// CHECK:   } to @start_loops_3{{$}}
// CHECK: p4hir.state @infinite_loop {
// CHECK:   p4hir.transition to @infinite_loop{{$}}
// CHECK: p4hir.state @finite_loop_7 {
// CHECK:   p4hir.assign {{.*}} : <!b2i>
// CHECK-NOT: p4corelib.extract_header
// CHECK:   p4hir.transition to @start_outOfBound_0{{$}}
// CHECK: p4hir.state @finite_loop_8 {
// CHECK:   p4hir.assign {{.*}} : <!b2i>
// CHECK-NOT: p4corelib.extract_header
// CHECK:   p4hir.transition to @start_outOfBound_0{{$}}
// CHECK: p4hir.state @finite_loop_9 {
// CHECK:   p4hir.assign {{.*}} : <!b2i>
// CHECK-NOT: p4corelib.extract_header
// CHECK:   p4hir.transition to @start_outOfBound_0{{$}}
// CHECK: p4hir.state @finite_loop_10 {
// CHECK:   p4hir.assign {{.*}} : <!b2i>
// CHECK-NOT: p4corelib.extract_header
// CHECK:   p4hir.transition to @start_outOfBound_0{{$}}
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
    bit<2> counter;
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
