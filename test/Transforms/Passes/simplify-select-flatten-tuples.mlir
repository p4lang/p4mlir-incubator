// RUN: p4mlir-opt --p4hir-simplify-select="flatten-tuples=true replace-bools=false replace-ranges=false concat-args=false" --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int34_b8i = #p4hir.int<34> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i
#set_const_of_true = #p4hir.set<const : [#true]> : !p4hir.set<!p4hir.bool>
#set_const_of_int1_b8i = #p4hir.set<const : [#int1_b8i]> : !p4hir.set<!b8i>
#set_const_of_int2_b8i = #p4hir.set<const : [#int2_b8i]> : !p4hir.set<!b8i>
#set_const_of_int3_b8i = #p4hir.set<const : [#int3_b8i]> : !p4hir.set<!b8i>

// Flatten all tuples in this select statement.
//   transition select(x, {{y}}, {y}, y, {x + 1, !y}) {
//     (1, {{false}}, {true}, w, {8w1, false}): reject;
//     (2, default, {true}, true, default): reject;
//     (3, {{true}}, default, true, {8w34, true}): reject;
//     default: accept;
//   }
// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.parser @p1
  p4hir.parser @p1(%arg0: !b8i, %arg1: !p4hir.bool)() {
    %set = p4hir.const #set_const_of_int3_b8i
    %set_0 = p4hir.const #set_const_of_true
    %set_1 = p4hir.const #set_const_of_int2_b8i
    %set_2 = p4hir.const #set_const_of_int1_b8i
    %c34_b8i = p4hir.const #int34_b8i
    %everything = p4hir.const #everything
    %true = p4hir.const #true
    %c1_b8i = p4hir.const #int1_b8i
    %false = p4hir.const #false
    %w = p4hir.variable ["w", init] : <!p4hir.bool>
    p4hir.assign %false, %w : <!p4hir.bool>

    // CHECK-LABEL: p4hir.state @start
    // CHECK-NOT: tuple
    p4hir.state @start {
      %tuple = p4hir.tuple (%arg1) : tuple<!p4hir.bool>
      %tuple_3 = p4hir.tuple (%tuple) : tuple<tuple<!p4hir.bool>>
      %tuple_4 = p4hir.tuple (%arg1) : tuple<!p4hir.bool>
      %add = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
      %not = p4hir.unary(not, %arg1) : !p4hir.bool
      %tuple_5 = p4hir.tuple (%add, %not) : tuple<!b8i, !p4hir.bool>
      p4hir.transition_select %arg0, %tuple_3, %tuple_4, %arg1, %tuple_5 : !b8i, tuple<tuple<!p4hir.bool>>, tuple<!p4hir.bool>, !p4hir.bool, tuple<!b8i, !p4hir.bool> {
        p4hir.select_case {
          %tuple_6 = p4hir.tuple (%false) : tuple<!p4hir.bool>
          %tuple_7 = p4hir.tuple (%tuple_6) : tuple<tuple<!p4hir.bool>>
          %set_8 = p4hir.set (%tuple_7) : !p4hir.set<tuple<tuple<!p4hir.bool>>>
          %tuple_9 = p4hir.tuple (%true) : tuple<!p4hir.bool>
          %set_10 = p4hir.set (%tuple_9) : !p4hir.set<tuple<!p4hir.bool>>
          %val = p4hir.read %w : <!p4hir.bool>
          %set_11 = p4hir.set (%val) : !p4hir.set<!p4hir.bool>
          %tuple_12 = p4hir.tuple (%c1_b8i, %false) : tuple<!b8i, !p4hir.bool>
          %set_13 = p4hir.set (%tuple_12) : !p4hir.set<tuple<!b8i, !p4hir.bool>>
          p4hir.yield %set_2, %set_8, %set_10, %set_11, %set_13 : !p4hir.set<!b8i>, !p4hir.set<tuple<tuple<!p4hir.bool>>>, !p4hir.set<tuple<!p4hir.bool>>, !p4hir.set<!p4hir.bool>, !p4hir.set<tuple<!b8i, !p4hir.bool>>
        } to @p1::@accept
        p4hir.select_case {
          %tuple_6 = p4hir.tuple (%true) : tuple<!p4hir.bool>
          %set_7 = p4hir.set (%tuple_6) : !p4hir.set<tuple<!p4hir.bool>>
          p4hir.yield %set_1, %everything, %set_7, %set_0, %everything : !p4hir.set<!b8i>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<tuple<!p4hir.bool>>, !p4hir.set<!p4hir.bool>, !p4hir.set<!p4hir.dontcare>
        } to @p1::@reject
        p4hir.select_case {
          %tuple_6 = p4hir.tuple (%true) : tuple<!p4hir.bool>
          %tuple_7 = p4hir.tuple (%tuple_6) : tuple<tuple<!p4hir.bool>>
          %set_8 = p4hir.set (%tuple_7) : !p4hir.set<tuple<tuple<!p4hir.bool>>>
          %tuple_9 = p4hir.tuple (%c34_b8i, %true) : tuple<!b8i, !p4hir.bool>
          %set_10 = p4hir.set (%tuple_9) : !p4hir.set<tuple<!b8i, !p4hir.bool>>
          p4hir.yield %set, %set_8, %everything, %set_0, %set_10 : !p4hir.set<!b8i>, !p4hir.set<tuple<tuple<!p4hir.bool>>>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.bool>, !p4hir.set<tuple<!b8i, !p4hir.bool>>
        } to @p1::@accept
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p1::@reject
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p1::@start
  }

  // CHECK-LABEL: p4hir.parser @p2
  p4hir.parser @p2(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "x"}, %arg1: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "y"}, %arg2: tuple<!b8i, !b8i> {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "w"})() {
    %everything = p4hir.const #everything
    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      %tuple = p4hir.tuple (%arg0, %arg1) : tuple<!b8i, !b8i>
      // CHECK: p4hir.transition_select
      p4hir.transition_select %tuple : tuple<!b8i, !b8i> {
        p4hir.select_case {
          // CHECK: %t0 = p4hir.tuple_extract %arg2[0] : tuple<!b8i, !b8i>
          // CHECK: %t1 = p4hir.tuple_extract %arg2[1] : tuple<!b8i, !b8i>
          %set = p4hir.set (%arg2) : !p4hir.set<tuple<!b8i, !b8i>>
          p4hir.yield %set : !p4hir.set<tuple<!b8i, !b8i>>
        } to @p2::@reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p2::@accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p2::@start
  }

  // CHECK-LABEL: p4hir.parser @p3
  p4hir.parser @p3(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "x"}, %arg1: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "y"}, %arg2: tuple<!b8i, !b8i> {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "w"})() {
    %everything = p4hir.const #everything
    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK: %t0 = p4hir.tuple_extract %arg2[0] : tuple<!b8i, !b8i>
      // CHECK: %t1 = p4hir.tuple_extract %arg2[1] : tuple<!b8i, !b8i>
      // CHECK: p4hir.transition_select
      p4hir.transition_select %arg2 : tuple<!b8i, !b8i> {
        p4hir.select_case {
          %tuple = p4hir.tuple (%arg0, %arg1) : tuple<!b8i, !b8i>
          %set = p4hir.set (%tuple) : !p4hir.set<tuple<!b8i, !b8i>>
          p4hir.yield %set : !p4hir.set<tuple<!b8i, !b8i>>
        } to @p3::@reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p3::@accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p3::@start
  }

}
