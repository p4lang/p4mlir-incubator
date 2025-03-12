// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b10i = !p4hir.bit<10>
!i16i = !p4hir.int<16>
!infint = !p4hir.infint
#int2_b10i = #p4hir.int<2> : !b10i
#int3_i16i = #p4hir.int<3> : !i16i
#int3_infint = #p4hir.int<3> : !infint
#int4_i16i = #p4hir.int<4> : !i16i
#int5_i16i = #p4hir.int<5> : !i16i
// CHECK: module
module {
  p4hir.control @Pipe1(%arg0: !b10i, %arg1: !i16i, %arg2: !p4hir.ref<!b10i>)() {
    p4hir.func action @foo() {
      %c3_i16i = p4hir.const #int3_i16i
      %add = p4hir.binop(add, %c3_i16i, %arg1) : !i16i
      %x1 = p4hir.variable ["x1", init] : <!i16i>
      p4hir.assign %add, %x1 : <!i16i>
      %c4_i16i = p4hir.const #int4_i16i
      %cast = p4hir.cast(%c4_i16i : !i16i) : !i16i
      p4hir.assign %cast, %x1 : <!i16i>
      p4hir.implicit_return
    }
    p4hir.func action @bar() {
      %c2_b10i = p4hir.const #int2_b10i
      %cast = p4hir.cast(%c2_b10i : !b10i) : !b10i
      %x1 = p4hir.variable ["x1", init] : <!b10i>
      p4hir.assign %cast, %x1 : <!b10i>
      %val = p4hir.read %x1 : <!b10i>
      %sub = p4hir.binop(sub, %val, %arg0) : !b10i
      p4hir.assign %sub, %x1 : <!b10i>
      %val_0 = p4hir.read %x1 : <!b10i>
      p4hir.assign %val_0, %arg2 : <!b10i>
      p4hir.implicit_return
    }
    p4hir.control_apply {
      %x1 = p4hir.variable ["x1", init] : <!b10i>
      p4hir.assign %arg0, %x1 : <!b10i>
      %c5_i16i = p4hir.const #int5_i16i
      %cast = p4hir.cast(%c5_i16i : !i16i) : !i16i
      %x2 = p4hir.variable ["x2", init] : <!i16i>
      p4hir.assign %cast, %x2 : <!i16i>
      p4hir.assign %arg1, %x2 : <!i16i>
      p4hir.call @bar () : () -> ()
      %c3 = p4hir.const #int3_infint
      %cast_0 = p4hir.cast(%c3 : !infint) : !i16i
      %eq = p4hir.cmp(eq, %arg1, %cast_0) : !i16i, !p4hir.bool
      p4hir.if %eq {
        p4hir.call @foo () : () -> ()
        %c3_i16i = p4hir.const #int3_i16i
        %cast_1 = p4hir.cast(%c3_i16i : !i16i) : !i16i
        p4hir.assign %cast_1, %x2 : <!i16i>
      }
    }
  }
}
