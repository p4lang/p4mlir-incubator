// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test5: extracts go to individual headers (no header stack),
// so there is nothing to unroll — the parser passes through unchanged.

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start
// CHECK: p4hir.state @access1
// CHECK: p4hir.state @access2
// CHECK: p4hir.state @access3
// CHECK-NOT: @access1_1
// CHECK-NOT: @access2_1

@__corelib
extern packet_in { void extract<T>(out T hdr); }

header ethernet_t { bit<16> etherType; }
header srcRoute_t { bit<15> port; }

struct headers {
    ethernet_t ethernet;
    srcRoute_t srcRoutes1;
    srcRoute_t srcRoutes2;
    srcRoute_t srcRoutes3;
}

parser p(packet_in packet, out headers hdr) {
    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            1:       access1;
            2:       access2;
            default: access3;
        }
    }
    state access1 { packet.extract(hdr.srcRoutes1); transition accept; }
    state access2 { packet.extract(hdr.srcRoutes2); transition access1; }
    state access3 { packet.extract(hdr.srcRoutes3); transition access2; }
}
