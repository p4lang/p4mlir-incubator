// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test10: a runtime (non-.next) stack index must be left as-is
// — no folding, no clones (p4c keeps hdr.hs[meta.hs_next_index] too).

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK:   p4hir.array_element_ref
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @start_1

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header data_t { bit<8> f; }
struct headers { data_t[4] hs; }

parser p(packet_in packet, out headers hdr, in bit<2> idx) {
    state start {
        packet.extract(hdr.hs[idx]);
        transition accept;
    }
}
