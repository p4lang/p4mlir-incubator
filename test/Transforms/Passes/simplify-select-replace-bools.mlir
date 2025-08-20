// RUN: p4mlir-opt --p4hir-simplify-select="flatten-tuples=false replace-bools=true replace-ranges=false concat-args=false" --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#set_const_of_false = #p4hir.set<const : [#false]> : !p4hir.set<!p4hir.bool>
#set_const_of_true = #p4hir.set<const : [#true]> : !p4hir.set<!p4hir.bool>
#set_const_of_int1_b8i = #p4hir.set<const : [#int1_b8i]> : !p4hir.set<!b8i>
#set_const_of_int2_b8i = #p4hir.set<const : [#int2_b8i]> : !p4hir.set<!b8i>

// CHECK-LABEL: module
module {
  p4hir.parser @p1(%arg0: !p4hir.bool, %arg1: !b8i, %arg2: !p4hir.bool)() {
    %set = p4hir.const #set_const_of_int2_b8i
    %set_0 = p4hir.const #set_const_of_int1_b8i
    %set_1 = p4hir.const #set_const_of_true
    %set_2 = p4hir.const #set_const_of_false
    %everything = p4hir.const #everything
    %false = p4hir.const #false
    %w = p4hir.variable ["w", init] : <!p4hir.bool>
    p4hir.assign %false, %w : <!p4hir.bool>

    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK: p4hir.transition_select %{{.*}} : !b1i, !b8i, !b1i {
      p4hir.transition_select %arg0, %arg1, %arg2 : !p4hir.bool, !b8i, !p4hir.bool {
        p4hir.select_case {
          // CHECK: p4hir.yield %{{.*}} : !p4hir.set<!b1i>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!b1i>
          p4hir.yield %set_2, %everything, %set_1 : !p4hir.set<!p4hir.bool>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          %val = p4hir.read %w : <!p4hir.bool>
          %set_3 = p4hir.set (%val) : !p4hir.set<!p4hir.bool>
          %val_4 = p4hir.read %w : <!p4hir.bool>
          %set_5 = p4hir.set (%val_4) : !p4hir.set<!p4hir.bool>
          // CHECK: p4hir.yield %{{.*}} : !p4hir.set<!b1i>, !p4hir.set<!b8i>, !p4hir.set<!b1i>
          p4hir.yield %set_3, %set_0, %set_5 : !p4hir.set<!p4hir.bool>, !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          %val = p4hir.read %w : <!p4hir.bool>
          %set_3 = p4hir.set (%val) : !p4hir.set<!p4hir.bool>
          // CHECK: p4hir.yield %{{.*}} : !p4hir.set<!b1i>, !p4hir.set<!b8i>, !p4hir.set<!b1i>
          p4hir.yield %set_1, %set, %set_3 : !p4hir.set<!p4hir.bool>, !p4hir.set<!b8i>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          // CHECK: p4hir.yield %{{.*}} : !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!b1i>
          p4hir.yield %everything, %everything, %set_2 : !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!p4hir.bool>
        } to @p1::@reject
        p4hir.select_case {
          // CHECK: p4hir.yield %{{.*}} : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p1::@accept
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
}

