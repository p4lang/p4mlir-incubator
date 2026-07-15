// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test8: a constant stack index hdr.h[0] is not a .next access
// and is kept verbatim — no folding beyond the literal, no clones.

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @parse_hdrs {
// CHECK:   p4hir.array_element_ref
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @parse_hdrs_1

@__corelib
extern packet_in { void extract<T>(out T hdr); }

header h_t { bit<8> f; }
struct headers { h_t[2] h; }

parser p(packet_in pkt, out headers hdr) {
    state start {
        transition parse_hdrs;
    }
    state parse_hdrs {
        pkt.extract(hdr.h[0]);
        transition accept;
    }
}
