// RUN: p4mlir-opt --p4hir-simplify-parsers %s | FileCheck %s

!b8i = !p4hir.bit<8>

#true = #p4hir.bool<true> : !p4hir.bool
#int5_b8i = #p4hir.int<5> : !b8i

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.parser @p
// CHECK:       p4hir.state @start
// CHECK:       p4hir.state @s0
// CHECK:       p4hir.transition_select
// CHECK:       p4hir.state @accept
// CHECK:       p4hir.state @reject

// CHECK-NOT:   p4hir.state @s1
// CHECK-NOT:   p4hir.state @s_dead
module {
  p4hir.parser @p()() {

    p4hir.state @start {
      p4hir.transition to @p::@s0
    }

    p4hir.state @s0 {
      %5 = p4hir.const #int5_b8i
      %v1 = p4hir.variable ["v1", init] : <!b8i>
      p4hir.assign %5, %v1 : <!b8i>
      p4hir.transition to @p::@s1
    }

    p4hir.state @s1 {
      %true = p4hir.const #true
      p4hir.transition_select %true : !p4hir.bool {
        p4hir.select_case {
          %set = p4hir.set (%true) : !p4hir.set<!p4hir.bool>
          p4hir.yield %set : !p4hir.set<!p4hir.bool>
        } to @p::@accept
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p::@reject

      }
    }

    p4hir.state @s_dead {
      p4hir.transition to @p::@accept
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


// CHECK-LABEL: module
// CHECK-LABEL: p4hir.parser @branching
// CHECK:       p4hir.state @start
// CHECK:       p4hir.state @s0
// CHECK:       p4hir.transition_select
// CHECK:       p4hir.state @s1
// CHECK:       p4hir.state @s2
// CHECK:       p4hir.state @accept

// CHECK-NOT:   p4hir.state @s_dead
// CHECK-NOT:   p4hir.state @reject
module {
  p4hir.parser @branching()() {
    p4hir.state @start {
      p4hir.transition to @branching::@s0
    }

    p4hir.state @s0 {
      %true = p4hir.const #true
      p4hir.transition_select %true : !p4hir.bool {
        p4hir.select_case {
          %set = p4hir.set (%true) : !p4hir.set<!p4hir.bool>
          p4hir.yield %set : !p4hir.set<!p4hir.bool>
        } to @branching::@s1
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @branching::@s2
      }
    }

    p4hir.state @s1 {
      p4hir.transition to @branching::@s2
    }

    p4hir.state @s2 {
      p4hir.transition to @branching::@accept
    }

    p4hir.state @s_dead {
      p4hir.transition to @branching::@accept
    }

    p4hir.state @accept {
      p4hir.parser_accept
    }

    p4hir.state @reject {
      p4hir.parser_reject
    }

    p4hir.transition to @branching::@start
  }
}
