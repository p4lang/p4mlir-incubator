// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

typedef bit<9> Narrow_t;
type Narrow_t Narrow;
typedef bit<32> Wide_t;
type Wide_t Wide;

// CHECK: !Narrow = !p4hir.alias<"Narrow", !b9i>
// CHECK: !Wide = !p4hir.alias<"Wide", !b32i>
// CHECK: #int10_b9i = #p4hir.int<10> : !b9i
// CHECK: #int3_b32i = #p4hir.int<3> : !b32i
// CHECK: #int192_Narrow = #p4hir.int<192> : !Narrow

// CHECK: %[[PSA_CPU_PORT:.*]] = p4hir.const ["PSA_CPU_PORT"] #int192_Narrow
const Narrow PSA_CPU_PORT = (Narrow) 9w192; // target-specific

// CHECK-LABEL: p4hir.func action @c
action c(out bool b) {
        Wide x = (Wide)3;
        Narrow y = (Narrow)(Narrow_t)(Wide_t)x;
// CHECK: %[[y:.*]] = p4hir.variable ["y", init] : <!Narrow>
// CHECK: p4hir.assign %[[PSA_CPU_PORT]], %[[y]] : <!Narrow>
        y = PSA_CPU_PORT;
        b = y == (Narrow)10;
}
