// RUN: p4mlir-opt %s | FileCheck %s

!empty = !p4hir.struct<"empty">
!i10i = !p4hir.int<10>
#false = #p4hir.bool<false> : !p4hir.bool
#subparser2_ctorArg = #p4hir.ctor_param<@subparser2, "ctorArg"> : !p4hir.bool
#int1_i10i = #p4hir.int<1> : !i10i
#int2_i10i = #p4hir.int<2> : !i10i
// CHECK: module
module {
  p4hir.parser @subparser(%arg0: !empty)() {
    p4hir.state @start {
      p4hir.transition to @subparser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @subparser::@start
  }
  p4hir.parser @subparser2(%arg0: !empty)(ctorArg: !p4hir.bool) {
    %ctorArg = p4hir.const ["ctorArg"] #subparser2_ctorArg
    p4hir.state @start {
      p4hir.transition to @subparser2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @subparser2::@start
  }
  p4hir.parser @p(%arg0: !empty, %arg1: !i10i)() {
    %s = p4hir.variable ["s", init] : <!i10i>
    p4hir.assign %arg1, %s : <!i10i>
    %sp = p4hir.instantiate @subparser() as "sp" : () -> !p4hir.parser<"subparser", (!empty)>
    %false = p4hir.const #false
    %sp2 = p4hir.instantiate @subparser2(%false) as "sp2" : (!p4hir.bool) -> !p4hir.parser<"subparser2", (!empty)>
    p4hir.state @start {
      %c1_i10i = p4hir.const #int1_i10i
      %cast = p4hir.cast(%c1_i10i : !i10i) : !i10i
      p4hir.assign %cast, %s : <!i10i>
      p4hir.transition to @p::@next
    }
    p4hir.state @next {
      %c2_i10i = p4hir.const #int2_i10i
      %cast = p4hir.cast(%c2_i10i : !i10i) : !i10i
      p4hir.assign %cast, %s : <!i10i>
      p4hir.transition to @p::@accept
    }
    p4hir.state @drop {
      p4hir.transition to @p::@reject
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p::@start
  }
}
