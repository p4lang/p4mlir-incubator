// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b1i = !p4hir.bit<1>
!b2i = !p4hir.bit<2>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b4i = !p4hir.bit<4>
!b7i = !p4hir.bit<7>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!infint = !p4hir.infint
#int-2_b2i = #p4hir.int<2> : !b2i
#int-6_b4i = #p4hir.int<10> : !b4i
#int0_b1i = #p4hir.int<0> : !b1i
#int0_b3i = #p4hir.int<0> : !b3i
#int0_b4i = #p4hir.int<0> : !b4i
#int42_infint = #p4hir.int<42> : !infint
// CHECK: module
module {
  p4hir.func action @foo(%arg0: !b32i {p4hir.dir = #p4hir<dir in>}, %arg1: !p4hir.ref<!i8i> {p4hir.dir = #p4hir<dir inout>}) {
    %0 = p4hir.slice %arg0[10 : 8] : !b32i -> !b3i
    %1 = p4hir.slice %0[2 : 1] : !b3i -> !b2i
    %b = p4hir.variable ["b", init] : <!b2i>
    p4hir.assign %1, %b : <!b2i>
    %2 = p4hir.slice_ref %arg1[7 : 1] : <!i8i> -> !b7i
    %d = p4hir.variable ["d", init] : <!b7i>
    p4hir.assign %2, %d : <!b7i>
    %e = p4hir.const ["e"] #int42_infint
    %c-2_b2i = p4hir.const #int-2_b2i
    %f = p4hir.variable ["f", init] : <!b2i>
    p4hir.assign %c-2_b2i, %f : <!b2i>
    %n = p4hir.variable ["n"] : <!b8i>
    %m = p4hir.variable ["m"] : <!b8i>
    %x = p4hir.variable ["x"] : <!b8i>
    %c0_b4i = p4hir.const #int0_b4i
    p4hir.assign_slice %c0_b4i, %n[7 : 4] : !b4i -> <!b8i>
    %c-6_b4i = p4hir.const #int-6_b4i
    %cast = p4hir.cast(%c-6_b4i : !b4i) : !b4i
    p4hir.assign_slice %cast, %m[7 : 4] : !b4i -> <!b8i>
    %c0_b3i = p4hir.const #int0_b3i
    p4hir.assign_slice %c0_b3i, %m[7 : 5] : !b3i -> <!b8i>
    %c0_b1i = p4hir.const #int0_b1i
    p4hir.assign_slice %c0_b1i, %x[5 : 5] : !b1i -> <!b8i>
    p4hir.return
  }
}
