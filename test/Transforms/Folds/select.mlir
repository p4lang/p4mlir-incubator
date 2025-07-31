// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int8_b8i = #p4hir.int<8> : !b8i
#set_const_of_false = #p4hir.set<const : [#false]> : !p4hir.set<!p4hir.bool>
#set_const_of_true = #p4hir.set<const : [#true]> : !p4hir.set<!p4hir.bool>
#set_const_of_int1_b8i = #p4hir.set<const : [#int1_b8i]> : !p4hir.set<!b8i>
#set_const_of_int2_b8i = #p4hir.set<const : [#int2_b8i]> : !p4hir.set<!b8i>
#set_const_of_int8_b8i = #p4hir.set<const : [#int8_b8i]> : !p4hir.set<!b8i>

// CHECK-LABEL: module
module {
  p4hir.parser @p1(%arg0: !b8i, %arg1: !p4hir.bool)() {
    %set = p4hir.const #set_const_of_int8_b8i
    %set_0 = p4hir.const #set_const_of_true
    %set_1 = p4hir.const #set_const_of_int2_b8i
    %set_2 = p4hir.const #set_const_of_false
    %set_3 = p4hir.const #set_const_of_int1_b8i
    %everything = p4hir.const #everything

    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK-COUNT-2: p4hir.select_case
      // CHECK-NOT: p4hir.yield %everything, %everything : !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>
      // CHECK: p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
      // Remove dead cases; canonicalize default cases.
      p4hir.transition_select %arg0, %arg1 : !b8i, !p4hir.bool {
        p4hir.select_case {
          p4hir.yield %set_3, %set_2 : !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %everything, %everything : !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>
        } to @p1::@next
        p4hir.select_case {
          p4hir.yield %set_1, %set_0 : !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
      }
    }

    // CHECK-LABEL: p4hir.state @next
    p4hir.state @next {
      // CHECK-NOT: p4hir.select_case
      // CHECK: p4hir.transition to @p1::@final
      // Replace select with one default case with direct transition.
      p4hir.transition_select %arg0, %arg1 : !b8i, !p4hir.bool {
        p4hir.select_case {
          p4hir.yield %everything, %everything : !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>
        } to @p1::@final
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %set, %set_2 : !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p1::@reject
      }
    }

    p4hir.state @final {
      p4hir.transition to @p1::@accept
    }

    p4hir.state @accept {
      p4hir.parser_accept
    }

    p4hir.state @reject {
      p4hir.parser_reject
    }

    p4hir.transition to @p1::@start
  }
}
