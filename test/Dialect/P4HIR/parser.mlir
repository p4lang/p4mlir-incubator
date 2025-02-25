// RUN: p4mlir-opt %s | FileCheck %s

!b10i = !p4hir.bit<10>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int0_b10i = #p4hir.int<0> : !b10i
#int10_b10i = #p4hir.int<10> : !b10i
#int1_b10i = #p4hir.int<1> : !b10i
#int20_b10i = #p4hir.int<20> : !b10i
// CHECK: module
module {
  p4hir.parser @p2(%arg0: !b10i, %arg1: !p4hir.ref<!p4hir.bool>) {
    p4hir.state @start {
      %true = p4hir.const #true
      %tuple = p4hir.tuple (%arg0, %true) : tuple<!b10i, !p4hir.bool>
      p4hir.transition_select %tuple : tuple<!b10i, !p4hir.bool> {
        p4hir.select_case {
          %c1_b10i = p4hir.const #int1_b10i
          %set = p4hir.set (%c1_b10i) : !p4hir.set<!b10i>
          %false = p4hir.const #false
          %set_0 = p4hir.set (%false) : !p4hir.set<!p4hir.bool>
          %setproduct = p4hir.set_product (%set, %set_0) : !p4hir.set<tuple<!b10i, !p4hir.bool>>
          p4hir.yield %setproduct : !p4hir.set<tuple<!b10i, !p4hir.bool>>
        } to @p2::@drop
        p4hir.select_case {
          %c10_b10i = p4hir.const #int10_b10i
          %c20_b10i = p4hir.const #int20_b10i
          %range = p4hir.range(%c10_b10i, %c20_b10i) : !p4hir.set<!b10i>
          %true_0 = p4hir.const #true
          %set = p4hir.set (%true_0) : !p4hir.set<!p4hir.bool>
          %setproduct = p4hir.set_product (%range, %set) : !p4hir.set<tuple<!b10i, !p4hir.bool>>
          p4hir.yield %setproduct : !p4hir.set<tuple<!b10i, !p4hir.bool>>
        } to @p2::@next
        p4hir.select_case {
          %c0_b10i = p4hir.const #int0_b10i
          %c0_b10i_0 = p4hir.const #int0_b10i
          %mask = p4hir.mask(%c0_b10i, %c0_b10i_0) : !p4hir.set<!b10i>
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          %setproduct = p4hir.set_product (%mask, %everything) : !p4hir.set<tuple<!b10i, !p4hir.dontcare>>
          p4hir.yield %setproduct : !p4hir.set<tuple<!b10i, !p4hir.dontcare>>
        } to @p2::@next
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          %everything_0 = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          %setproduct = p4hir.set_product (%everything, %everything_0) : !p4hir.set<tuple<!p4hir.dontcare, !p4hir.dontcare>>
          p4hir.yield %setproduct : !p4hir.set<tuple<!p4hir.dontcare, !p4hir.dontcare>>
        } to @p2::@reject
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p2::@reject
      }
    }
    p4hir.state @drop {
      p4hir.transition to @p2::@reject
    }
    p4hir.state @next {
      %true = p4hir.const #true
      p4hir.assign %true, %arg1 : <!p4hir.bool>
      p4hir.transition to @p2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p2::@start
  }
}
