// RUN: p4mlir-opt %s | FileCheck %s
// Test the P4HIR operations can parse and print correctly (roundtrip)

!bit32 = !p4hir.bit<32>
!int64 = !p4hir.int<64>

module  {
   %0 = p4hir.variable ["tmp"] : <!bit32>
   %1 = p4hir.variable ["foo", init] : <!int64>
   %2 = p4hir.variable ["bar"] : <!p4hir.infint>
}

// CHECK: module {
// CHECK-NEXT: p4hir.variable ["tmp"] : <!b32i>
// CHECK-NEXT: p4hir.variable ["foo", init] : <!i64i>
// CHECK-NEXT: p4hir.variable ["bar"] : <!infint>
// CHECK-NEXT: }
