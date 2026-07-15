// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt -lower-to-p4corelib -p4hir-parser-unroll | FileCheck %s
// p4c parser-unroll-test6: a single .next extract folds to index 0; no clones.

// CHECK: p4hir.parser @p
// CHECK: p4hir.state @start {
// CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
// CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
// CHECK:   p4hir.transition to @accept
// CHECK-NOT: @start_1

@__corelib
extern packet_in { void extract<T>(out T hdr); }

header test_t { bit<8> f; }
struct headers { test_t[4] test; }

parser p(packet_in b, out headers hdr) {
    state start {
        b.extract(hdr.test.next);
        transition accept;
    }
}
