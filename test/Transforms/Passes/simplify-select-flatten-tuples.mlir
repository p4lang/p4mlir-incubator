// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --p4hir-select-flatten-tuples --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
!i32i = !p4hir.int<8>
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
#int1_i32i = #p4hir.int<1> : !i32i
#int2_i32i = #p4hir.int<2> : !i32i
#int3_i32i = #p4hir.int<3> : !i32i
#int4_i32i = #p4hir.int<4> : !i32i
#set_const_of_ = #p4hir.set<const : [#p4hir.aggregate<[#int4_i32i]> : tuple<!i32i>]> : !p4hir.set<tuple<!i32i>>
#set_const_of_1 = #p4hir.set<const : [#p4hir.aggregate<[#int1_i32i, #int2_i32i, #int3_i32i]> : tuple<!i32i, !i32i, !i32i>]> : !p4hir.set<tuple<!i32i, !i32i, !i32i>>

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

    // CHECK:      %[[SET_TRUE:.*]] = p4hir.const #set_const_of_true
    // CHECK-NEXT: %[[SET_FALSE:.*]] = p4hir.const #set_const_of_false
    // CHECK-NEXT: %[[SET_3:.*]] = p4hir.const #set_const_of_int3_b8i
    // CHECK-NEXT: %[[SET_2:.*]] = p4hir.const #set_const_of_int2_b8i
    // CHECK-NEXT: %[[SET_1:.*]] = p4hir.const #set_const_of_int1_b8i
    // CHECK-NEXT: %everything = p4hir.const #everything
    // CHECK-NEXT: %c1_b8i = p4hir.const #int1_b8i
    // CHECK-NEXT: p4hir.state @start {
    // CHECK-NEXT:   %[[ADD:.*]] = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
    // CHECK-NEXT:   %[[NOT:.*]] = p4hir.unary(not, %arg1) : !p4hir.bool
    // CHECK-NEXT:   p4hir.transition_select %arg0, %arg1, %arg1, %arg1, %[[ADD]], %[[NOT]] : !b8i, !p4hir.bool, !p4hir.bool, !p4hir.bool, !b8i, !p4hir.bool {
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %[[SET_1]], %[[SET_FALSE]] : !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
    // CHECK-NEXT:     } to @accept
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %[[SET_2]], %everything : !p4hir.set<!b8i>, !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @reject
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %[[SET_3]], %[[SET_TRUE]] : !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
    // CHECK-NEXT:     } to @accept
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @reject
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    p4hir.state @start {
      %tuple = p4hir.tuple(%arg1) : tuple<!p4hir.bool>
      %tuple_3 = p4hir.tuple(%tuple) : tuple<tuple<!p4hir.bool>>
      %tuple_4 = p4hir.tuple(%arg1) : tuple<!p4hir.bool>
      %add = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
      %not = p4hir.unary(not, %arg1) : !p4hir.bool
      %tuple_5 = p4hir.tuple(%add, %not) : tuple<!b8i, !p4hir.bool>
      p4hir.transition_select %arg0, %tuple_3, %tuple_4, %arg1, %tuple_5 : !b8i, tuple<tuple<!p4hir.bool>>, tuple<!p4hir.bool>, !p4hir.bool, tuple<!b8i, !p4hir.bool> {
        p4hir.select_case {
          %tuple_6 = p4hir.tuple(%false) : tuple<!p4hir.bool>
          %tuple_7 = p4hir.tuple(%tuple_6) : tuple<tuple<!p4hir.bool>>
          %set_8 = p4hir.set (%tuple_7) : !p4hir.set<tuple<tuple<!p4hir.bool>>>
          %tuple_9 = p4hir.tuple(%true) : tuple<!p4hir.bool>
          %set_10 = p4hir.set (%tuple_9) : !p4hir.set<tuple<!p4hir.bool>>
          %val = p4hir.read %w : <!p4hir.bool>
          %set_11 = p4hir.set (%val) : !p4hir.set<!p4hir.bool>
          %tuple_12 = p4hir.tuple(%c1_b8i, %false) : tuple<!b8i, !p4hir.bool>
          %set_13 = p4hir.set (%tuple_12) : !p4hir.set<tuple<!b8i, !p4hir.bool>>
          p4hir.yield %set_2, %set_8, %set_10, %set_11, %set_13 : !p4hir.set<!b8i>, !p4hir.set<tuple<tuple<!p4hir.bool>>>, !p4hir.set<tuple<!p4hir.bool>>, !p4hir.set<!p4hir.bool>, !p4hir.set<tuple<!b8i, !p4hir.bool>>
        } to @accept
        p4hir.select_case {
          %tuple_6 = p4hir.tuple(%true) : tuple<!p4hir.bool>
          %set_7 = p4hir.set (%tuple_6) : !p4hir.set<tuple<!p4hir.bool>>
          p4hir.yield %set_1, %everything, %set_7, %set_0, %everything : !p4hir.set<!b8i>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<tuple<!p4hir.bool>>, !p4hir.set<!p4hir.bool>, !p4hir.set<!p4hir.dontcare>
        } to @reject
        p4hir.select_case {
          %tuple_6 = p4hir.tuple(%true) : tuple<!p4hir.bool>
          %tuple_7 = p4hir.tuple(%tuple_6) : tuple<tuple<!p4hir.bool>>
          %set_8 = p4hir.set (%tuple_7) : !p4hir.set<tuple<tuple<!p4hir.bool>>>
          %tuple_9 = p4hir.tuple(%c34_b8i, %true) : tuple<!b8i, !p4hir.bool>
          %set_10 = p4hir.set (%tuple_9) : !p4hir.set<tuple<!b8i, !p4hir.bool>>
          p4hir.yield %set, %set_8, %everything, %set_0, %set_10 : !p4hir.set<!b8i>, !p4hir.set<tuple<tuple<!p4hir.bool>>>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.bool>, !p4hir.set<tuple<!b8i, !p4hir.bool>>
        } to @accept
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @reject
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @p2
  p4hir.parser @p2(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "x"}, %arg1: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "y"}, %arg2: tuple<!b8i, !b8i> {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "w"})() {
    %everything = p4hir.const #everything

    // CHECK:      %everything = p4hir.const #everything
    // CHECK-NEXT: p4hir.state @start {
    // CHECK-NEXT:   p4hir.transition_select %arg0, %arg1 : !b8i, !b8i {
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       %[[T0:.*]] = p4hir.tuple_extract %arg2[0] : tuple<!b8i, !b8i>
    // CHECK-NEXT:       %[[SET_T0:.*]] = p4hir.set (%[[T0]]) : !p4hir.set<!b8i>
    // CHECK-NEXT:       %[[T1:.*]] = p4hir.tuple_extract %arg2[1] : tuple<!b8i, !b8i>
    // CHECK-NEXT:       %[[SET_T1:.*]] = p4hir.set (%[[T1]]) : !p4hir.set<!b8i>
    // CHECK-NEXT:       p4hir.yield %[[SET_T0]], %[[SET_T1]] : !p4hir.set<!b8i>, !p4hir.set<!b8i>
    // CHECK-NEXT:     } to @reject
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @accept
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    p4hir.state @start {
      %tuple = p4hir.tuple(%arg0, %arg1) : tuple<!b8i, !b8i>
      p4hir.transition_select %tuple : tuple<!b8i, !b8i> {
        p4hir.select_case {
          %set = p4hir.set (%arg2) : !p4hir.set<tuple<!b8i, !b8i>>
          p4hir.yield %set : !p4hir.set<tuple<!b8i, !b8i>>
        } to @reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  // CHECK-LABEL: p4hir.parser @p3
  p4hir.parser @p3(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "x"}, %arg1: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "y"}, %arg2: tuple<!b8i, !b8i> {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "w"})() {
    %everything = p4hir.const #everything

    // CHECK:      %everything = p4hir.const #everything
    // CHECK-NEXT: p4hir.state @start {
    // CHECK-NEXT:   %[[T0:.*]] = p4hir.tuple_extract %arg2[0] : tuple<!b8i, !b8i>
    // CHECK-NEXT:   %[[T1:.*]] = p4hir.tuple_extract %arg2[1] : tuple<!b8i, !b8i>
    // CHECK-NEXT:   p4hir.transition_select %[[T0]], %[[T1]] : !b8i, !b8i {
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       %[[SET_ARG0:.*]] = p4hir.set (%arg0) : !p4hir.set<!b8i>
    // CHECK-NEXT:       %[[SET_ARG1:.*]] = p4hir.set (%arg1) : !p4hir.set<!b8i>
    // CHECK-NEXT:       p4hir.yield %[[SET_ARG0]], %[[SET_ARG1]] : !p4hir.set<!b8i>, !p4hir.set<!b8i>
    // CHECK-NEXT:     } to @reject
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @accept
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    p4hir.state @start {
      p4hir.transition_select %arg2 : tuple<!b8i, !b8i> {
        p4hir.select_case {
          %tuple = p4hir.tuple(%arg0, %arg1) : tuple<!b8i, !b8i>
          %set = p4hir.set (%tuple) : !p4hir.set<tuple<!b8i, !b8i>>
          p4hir.yield %set : !p4hir.set<tuple<!b8i, !b8i>>
        } to @reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }

  p4hir.parser @weird(%arg0: !i32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg1"}, %arg1: !i32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg2"})() {
    %set = p4hir.const #set_const_of_
    %set_0 = p4hir.const #set_const_of_1
    %everything = p4hir.const #everything

    // CHECK:      %[[SET_4:.*]] = p4hir.const #set_const_of_int4_i8i
    // CHECK-NEXT: %[[SET_3:.*]] = p4hir.const #set_const_of_int3_i8i
    // CHECK-NEXT: %[[SET_2:.*]] = p4hir.const #set_const_of_int2_i8i
    // CHECK-NEXT: %[[SET_1:.*]] = p4hir.const #set_const_of_int1_i8i
    // CHECK-NEXT: %everything = p4hir.const #everything
    // CHECK-NEXT: p4hir.state @start {
    // CHECK-NEXT:   p4hir.transition_select %arg0, %arg1, %arg1, %arg1 : !i8i, !i8i, !i8i, !i8i {
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       %[[SET_ARG0:.*]] = p4hir.set (%arg0) : !p4hir.set<!i8i>
    // CHECK-NEXT:       p4hir.yield %[[SET_ARG0]], %[[SET_1]], %[[SET_2]], %[[SET_3]] : !p4hir.set<!i8i>, !p4hir.set<!i8i>, !p4hir.set<!i8i>, !p4hir.set<!i8i>
    // CHECK-NEXT:     } to @foo1
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %[[SET_4]], %everything, %everything, %everything : !p4hir.set<!i8i>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @reject
    // CHECK-NEXT:     p4hir.select_case {
    // CHECK-NEXT:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:     } to @accept
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    p4hir.state @start {
      %tuple = p4hir.tuple(%arg0) : tuple<!i32i>
      %tuple_1 = p4hir.tuple(%arg1, %arg1, %arg1) : tuple<!i32i, !i32i, !i32i>
      p4hir.transition_select %tuple, %tuple_1 : tuple<!i32i>, tuple<!i32i, !i32i, !i32i> {
        p4hir.select_case {
          %tuple_2 = p4hir.tuple(%arg0) : tuple<!i32i>
          %set_3 = p4hir.set (%tuple_2) : !p4hir.set<tuple<!i32i>>
          p4hir.yield %set_3, %set_0 : !p4hir.set<tuple<!i32i>>, !p4hir.set<tuple<!i32i, !i32i, !i32i>>
        } to @foo1
        p4hir.select_case {
          p4hir.yield %set, %everything : !p4hir.set<tuple<!i32i>>, !p4hir.set<!p4hir.dontcare>
        } to @reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @accept
      }
    }
    p4hir.state @foo1 {
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }
}
