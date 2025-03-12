// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b32i = !p4hir.bit<32>
!b9i = !p4hir.bit<9>
!Narrow = !p4hir.alias<"Narrow", !b9i>
!Wide = !p4hir.alias<"Wide", !b32i>
#int10_b9i = #p4hir.int<10> : !b9i
#int3_b32i = #p4hir.int<3> : !b32i
#int192_Narrow = #p4hir.int<192> : !Narrow
// CHECK: module
module {
  %PSA_CPU_PORT = p4hir.const ["PSA_CPU_PORT"] #int192_Narrow
}
