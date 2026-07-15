// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test7: two hdr.*.next accesses in ONE state take
// consecutive constant indices 0 and 1 (not both 0).

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
// CHECK:   %[[C1:.*]] = p4hir.const #int1_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[C1]]]
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @start_1

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header data_t { bit<8> f; }
struct headers { data_t[2] hs; }

parser p(packet_in packet, out headers hdr) {
    state start {
        packet.extract(hdr.hs.next);
        packet.extract(hdr.hs.next);
        transition accept;
    }
}
