// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// Loop-less .next chain (p4c parser-unroll-test4): access3 -> access2 -> access1
// -> last, entered at three points. Each reachable (state,index) folds to a
// constant; the index multiset is {0,0,0,1,1,2}, matching p4c.

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @last {
// CHECK:   p4hir.transition to @accept
// CHECK: p4hir.state @access1 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[A0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[A0]]]
// CHECK:   p4hir.transition to @last
// CHECK: p4hir.state @access1_1 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[A1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[A1]]]
// CHECK:   p4hir.transition to @last
// CHECK: p4hir.state @access1_2 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[A2:.*]] = p4hir.const #int2_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[A2]]]
// CHECK:   p4hir.transition to @last
// CHECK: p4hir.state @access2 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[B0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[B0]]]
// CHECK:   p4hir.transition to @access1_1
// CHECK: p4hir.state @access2_1 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[B1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[B1]]]
// CHECK:   p4hir.transition to @access1_2
// CHECK: p4hir.state @access3 {
// CHECK:   p4hir.binop(add
// CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
// CHECK:   p4hir.transition to @access2_1
// CHECK-NOT: @access1_3
// CHECK-NOT: @access3_1

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header ethernet_t { bit<16> etherType; }
header srcRoute_t { bit<1> bos; bit<15> port; }

struct headers {
    ethernet_t        ethernet;
    srcRoute_t[4]     srcRoutes;
    bit<32>           index;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        hdr.index = 0;
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0:       last;
            1:       access1;
            2:       access2;
            3:       access3;
            default: accept;
        }
    }
    state last {
        hdr.index = hdr.index + 1;
        transition accept;
    }
    state access1 {
        hdr.index = hdr.index + 1;
        packet.extract(hdr.srcRoutes.next);
        transition last;
    }
    state access2 {
        hdr.index = hdr.index + 1;
        packet.extract(hdr.srcRoutes.next);
        transition access1;
    }
    state access3 {
        hdr.index = hdr.index + 1;
        packet.extract(hdr.srcRoutes.next);
        transition access2;
    }
}
