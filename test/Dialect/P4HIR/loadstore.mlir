// RUN: p4mlir-opt %s | FileCheck %s

// Test the P4HIR operations can parse and print correctly (roundtrip)

!bit32 = !p4hir.bit<32>
!int64 = !p4hir.int<64>

module  {
   %0 = p4hir.variable ["tmp"] : <!bit32>
   %1 = p4hir.variable ["foo", init] : <!int64>
   %2 = p4hir.variable ["bar"] : <!p4hir.infint>

   %5 = p4hir.read %0 : <!bit32>
   %6 = p4hir.const #p4hir.int<65535> : !int64
   p4hir.assign %6, %1 : <!int64>
}

// CHECK: module {
// CHECK-NEXT: %[[VAL_0:.*]] = p4hir.variable ["tmp"] : <!p4hir.bit<32>>
// CHECK-NEXT: %[[VAL_1:.*]] = p4hir.variable ["foo", init] : <!p4hir.int<64>>
// CHECK-NEXT: p4hir.variable ["bar"] : <!p4hir.infint>
// CHECK-NEXT: p4hir.read %[[VAL_0]] : <!p4hir.bit<32>>
// CHECK-NEXT: %[[CONST_0:.*]] = p4hir.const #p4hir.int<65535> : !p4hir.int<64>
// CHECK-NEXT: p4hir.assign %[[CONST_0]], %[[VAL_1]] : <!p4hir.int<64>>
// CHECK-NEXT: }
