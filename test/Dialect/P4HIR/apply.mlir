// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b10i = !p4hir.bit<10>
!i16i = !p4hir.int<16>
!infint = !p4hir.infint
#int1_b10i = #p4hir.int<1> : !b10i
#int2_infint = #p4hir.int<2> : !infint
!InnerPipe = !p4hir.control<"InnerPipe", (!b10i, !i16i, !p4hir.ref<!i16i>)>
// CHECK: module
module {
  p4hir.control @InnerPipe(%arg0: !b10i, %arg1: !i16i, %arg2: !p4hir.ref<!i16i>)() {
    p4hir.control_apply {
    }
  }
  p4hir.control @Pipe(%arg0: !b10i, %arg1: !i16i, %arg2: !p4hir.ref<!i16i>, %arg3: !p4hir.ref<!i16i>)() {
    %inner = p4hir.instantiate @InnerPipe() as "inner" : () -> !InnerPipe
    p4hir.func action @bar() {
      %x1 = p4hir.variable ["x1"] : <!i16i>
      p4hir.return
    }
    p4hir.control_apply {
      p4hir.call @bar () : () -> ()
      %x1 = p4hir.variable ["x1"] : <!i16i>
      p4hir.scope {
        %c1_b10i = p4hir.const #int1_b10i
        %c2 = p4hir.const #int2_infint
        %cast = p4hir.cast(%c2 : !infint) : !i16i
        %arg3_out_arg = p4hir.variable ["arg3_out_arg"] : <!i16i>
        p4hir.apply %inner(%c1_b10i, %cast, %arg3_out_arg) : !InnerPipe
        %val = p4hir.read %arg3_out_arg : <!i16i>
        p4hir.assign %val, %x1 : <!i16i>
      }
    }
  }
}
