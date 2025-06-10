// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b10i = !p4hir.bit<10>
!i16i = !p4hir.int<16>
!infint = !p4hir.infint
!type_U = !p4hir.type_var<"U">
#InnerPipe_flag = #p4hir.ctor_param<@InnerPipe, "flag"> : !p4hir.bool
#false = #p4hir.bool<false> : !p4hir.bool
#in = #p4hir<dir in>
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>
#int10_b10i = #p4hir.int<10> : !b10i
#int2_b10i = #p4hir.int<2> : !b10i
#int3_i16i = #p4hir.int<3> : !i16i
#int3_infint = #p4hir.int<3> : !infint
#int42_i16i = #p4hir.int<42> : !i16i
#int42_infint = #p4hir.int<42> : !infint
#int4_i16i = #p4hir.int<4> : !i16i
#int5_i16i = #p4hir.int<5> : !i16i

// CHECK: module
module {
  p4hir.control @InnerPipe(%arg0: !b10i {p4hir.dir = #undir, p4hir.param_name = "arg1"}, %arg1: !i16i {p4hir.dir = #in, p4hir.param_name = "arg2"}, %arg2: !p4hir.ref<!i16i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "arg3"})(flag: !p4hir.bool) {
    %flag = p4hir.const ["flag"] #InnerPipe_flag
    p4hir.control_local @arg1 = %arg0 : !b10i
    p4hir.control_local @arg2 = %arg1 : !i16i
    p4hir.control_local @arg3 = %arg2 : !p4hir.ref<!i16i>
    p4hir.control_apply {
    }
  }
  p4hir.control @Pipe1(%arg0: !b10i {p4hir.dir = #undir, p4hir.param_name = "arg1"}, %arg1: !i16i {p4hir.dir = #in, p4hir.param_name = "arg2"}, %arg2: !p4hir.ref<!b10i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "oarg2"})() {
    p4hir.control_local @arg1 = %arg0 : !b10i
    p4hir.control_local @arg2 = %arg1 : !i16i
    p4hir.control_local @oarg2 = %arg2 : !p4hir.ref<!b10i>
    %topC = p4hir.const ["topC"] #int42_infint
    %c10_b10i = p4hir.const #int10_b10i
    %cast = p4hir.cast(%c10_b10i : !b10i) : !b10i
    %topV = p4hir.variable ["topV", init] : <!b10i>
    p4hir.assign %cast, %topV : <!b10i>
    p4hir.control_local @topV = %topV : !p4hir.ref<!b10i>
    %true = p4hir.const #true
    p4hir.instantiate @InnerPipe (%true : !p4hir.bool) as @inner1
    %false = p4hir.const #false
    p4hir.instantiate @InnerPipe (%false : !p4hir.bool) as @inner2
    %c42 = p4hir.const #int42_infint
    p4hir.func action @foo() {
      %c3_i16i = p4hir.const #int3_i16i
      %arg2_0 = p4hir.symbol_ref @arg2 : !i16i
      %add = p4hir.binop(add, %c3_i16i, %arg2_0) : !i16i
      %c42_i16i = p4hir.const #int42_i16i
      %add_1 = p4hir.binop(add, %add, %c42_i16i) : !i16i
      %x1 = p4hir.variable ["x1", init] : <!i16i>
      p4hir.assign %add_1, %x1 : <!i16i>
      %c4_i16i = p4hir.const #int4_i16i
      %val = p4hir.read %x1 : <!i16i>
      p4hir.assign %add, %x1 : <!i16i>
      p4hir.return
    }
    p4hir.func action @bar() {
      %c2_b10i = p4hir.const #int2_b10i
      %topV_0 = p4hir.symbol_ref @topV : !p4hir.ref<!b10i>
      %val = p4hir.read %topV_0 : <!b10i>
      %add = p4hir.binop(add, %c2_b10i, %val) : !b10i
      %x1 = p4hir.variable ["x1", init] : <!b10i>
      p4hir.assign %add, %x1 : <!b10i>
      %val_1 = p4hir.read %x1 : <!b10i>
      %arg1_2 = p4hir.symbol_ref @arg1 : !b10i
      %sub = p4hir.binop(sub, %val_1, %arg1_2) : !b10i
      p4hir.assign %sub, %x1 : <!b10i>
      %oarg2 = p4hir.symbol_ref @oarg2 : !p4hir.ref<!b10i>
      %val_3 = p4hir.read %x1 : <!b10i>
      p4hir.assign %val_3, %oarg2 : <!b10i>
      %topV_4 = p4hir.symbol_ref @topV : !p4hir.ref<!b10i>
      %val_5 = p4hir.read %x1 : <!b10i>
      p4hir.assign %val_5, %topV_4 : <!b10i>
      p4hir.return
    }
    p4hir.control_apply {
      %x1 = p4hir.variable ["x1", init] : <!b10i>
      p4hir.assign %arg0, %x1 : <!b10i>
      %c5_i16i = p4hir.const #int5_i16i
      %cast_0 = p4hir.cast(%c5_i16i : !i16i) : !i16i
      %x2 = p4hir.variable ["x2", init] : <!i16i>
      p4hir.assign %cast_0, %x2 : <!i16i>
      p4hir.assign %arg1, %x2 : <!i16i>
      p4hir.call @Pipe1::@bar () : () -> ()
      %c3 = p4hir.const #int3_infint
      %cast_1 = p4hir.cast(%c3 : !infint) : !i16i
      %eq = p4hir.cmp(eq, %arg1, %cast_1) : !i16i, !p4hir.bool
      p4hir.if %eq {
        p4hir.call @Pipe1::@foo () : () -> ()
        %c3_i16i = p4hir.const #int3_i16i
        %cast_2 = p4hir.cast(%c3_i16i : !i16i) : !i16i
        p4hir.assign %cast_2, %x2 : <!i16i>
      }
    }
  }
}
