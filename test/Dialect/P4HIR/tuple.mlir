// RUN: p4mlir-opt %s | FileCheck %s

!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
#int10_b32i = #p4hir.int<10> : !b32i
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#false = #p4hir.bool<false> : !p4hir.bool
#int12_b16i = #p4hir.int<12> : !b16i

// CHECK: module
module {
  %t = p4hir.const ["t"] #p4hir.aggregate<[#int0_b32i, #int1_b32i]> : tuple<!b32i, !b32i>

  p4hir.func action @test(%arg0: !p4hir.ref<!b16i> {p4hir.dir = #p4hir<dir out>}) {
    %c10_b32i = p4hir.const #int10_b32i
    %c12_b16i = p4hir.const #int12_b16i
    %tuple = p4hir.tuple (%c10_b32i, %c12_b16i) : tuple<!b32i, !b16i>
    %x_0 = p4hir.variable ["x", init] : <tuple<!b32i, !b16i>>
    p4hir.assign %tuple, %x_0 : <tuple<!b32i, !b16i>>
    %val_4 = p4hir.read %x_0 : <tuple<!b32i, !b16i>>
    %t1 = p4hir.tuple_extract %val_4[1] : tuple<!b32i, !b16i>
    p4hir.assign %t1, %arg0 : <!b16i>

    p4hir.implicit_return
  }

  p4hir.func action @test2() {
    %c10_b32i = p4hir.const #int10_b32i
    %false = p4hir.const #false
    %tuple = p4hir.tuple (%c10_b32i, %false) : tuple<!b32i, !p4hir.bool>
    %x_0 = p4hir.variable ["x", init] : <tuple<!b32i, !p4hir.bool>>
    p4hir.assign %tuple, %x_0 : <tuple<!b32i, !p4hir.bool>>
    %y = p4hir.variable ["y"] : <tuple<!b32i, !p4hir.bool>>
    %val = p4hir.read %x_0 : <tuple<!b32i, !p4hir.bool>>
    p4hir.assign %val, %y : <tuple<!b32i, !p4hir.bool>>
    p4hir.implicit_return
  }
}
