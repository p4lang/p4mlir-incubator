// RUN: p4mlir-opt %s | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>
!S = !p4hir.struct<"S", s1: !T, s2: !T>
!Empty = !p4hir.struct<"Empty">
!b9i = !p4hir.bit<9>
!PortId_t = !p4hir.struct<"PortId_t", _v: !b9i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i
#int1_b9i = #p4hir.int<1> : !b9i

// CHECK: module
module {
  %e = p4hir.const ["e"] #p4hir.aggregate<[]> : !Empty
  %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T

  p4hir.func action @test2(%arg0: !p4hir.ref<!PortId_t> {p4hir.dir = #p4hir<dir inout>}) {
    %_v = p4hir.struct_extract_ref %arg0["_v"] : <!PortId_t>
    %val = p4hir.read %arg0 : <!PortId_t>
    %_v_0 = p4hir.struct_extract %val["_v"] : !PortId_t
    %c1_b9i = p4hir.const #int1_b9i
    %add = p4hir.binop(add, %_v_0, %c1_b9i) : !b9i
    p4hir.assign %add, %_v : <!b9i>
    p4hir.return
  }

  p4hir.func action @test() {
    %vv = p4hir.variable ["vv"] : <!b9i>
    %val = p4hir.read %vv : <!b9i>
    %0 = p4hir.struct (%val) : !PortId_t
    %p1 = p4hir.variable ["p1", init] : <!PortId_t>
    p4hir.assign %0, %p1 : <!PortId_t>
    p4hir.return    
  }
}
