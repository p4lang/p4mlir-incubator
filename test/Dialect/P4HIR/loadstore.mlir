// RUN: p4mlir-opt %s | FileCheck %s

// Test the P4HIR operations can parse and print correctly (roundtrip)

!bit32 = !p4hir.bit<32>
!int64 = !p4hir.int<64>

module  {
   %0 = p4hir.alloca !bit32 ["tmp"] : !p4hir.ref<!bit32>
   %1 = p4hir.alloca !int64 ["foo", init] : !p4hir.ref<!int64>
   %2 = p4hir.alloca !p4hir.infint ["bar"] : !p4hir.ref<!p4hir.infint>

   %5 = p4hir.load %0 : !p4hir.ref<!bit32>, !bit32
   %6 = p4hir.const #p4hir.int<65535> : !int64
   p4hir.store %6, %1 : !int64, !p4hir.ref<!int64>
}

// CHECK: module {
// CHECK-NEXT: %[[VAL_0:.*]] = p4hir.alloca !p4hir.bit<32> ["tmp"] : !p4hir.ref<!p4hir.bit<32>>
// CHECK-NEXT: %[[VAL_1:.*]] = p4hir.alloca !p4hir.int<64> ["foo", init] : !p4hir.ref<!p4hir.int<64>>
// CHECK-NEXT: p4hir.alloca !p4hir.infint ["bar"] : !p4hir.ref<!p4hir.infint>
// CHECK-NEXT: p4hir.load %[[VAL_0]] : !p4hir.ref<!p4hir.bit<32>>, !p4hir.bit<32>
// CHECK-NEXT: %[[CONST_0:.*]] = p4hir.const #p4hir.int<65535> : !p4hir.int<64>
// CHECK-NEXT: p4hir.store %[[CONST_0]], %[[VAL_1]] : !p4hir.int<64>, !p4hir.ref<!p4hir.int<64>>
// CHECK-NEXT: }
