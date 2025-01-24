// RUN: p4mlir-opt %s | FileCheck %s
// Test the P4HIR operations can parse and print correctly (roundtrip)

!bit32 = !p4hir.bit<32>
!int64 = !p4hir.int<64>

module  {
   %0 = p4hir.alloca !bit32 ["tmp"] : !p4hir.ref<!bit32>
   %1 = p4hir.alloca !int64 ["foo", init] : !p4hir.ref<!int64>
   %2 = p4hir.alloca !p4hir.infint ["bar"] : !p4hir.ref<!p4hir.infint>
}

// CHECK: module {
// CHECK-NEXT: p4hir.alloca !p4hir.bit<32> ["tmp"] : !p4hir.ref<!p4hir.bit<32>>
// CHECK-NEXT: p4hir.alloca !p4hir.int<64> ["foo", init] : !p4hir.ref<!p4hir.int<64>>
// CHECK-NEXT: p4hir.alloca !p4hir.infint ["bar"] : !p4hir.ref<!p4hir.infint>
// CHECK-NEXT: }
