// RUN: p4mlir-opt --p4hir-remove-parser-control-flow --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#int10_b8i = #p4hir.int<10> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.parser @p1
  p4hir.parser @p1(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i

    // CHECK: p4hir.state @start_pre_0 {
    // CHECK:   %eq = p4hir.cmp(eq, %arg0, %c2_b8i) : !b8i, !p4hir.bool
    // CHECK:   p4hir.transition_select %eq : !p4hir.bool {
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %{{.*}}
    // CHECK:     } to @p1::@start_then_0
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK:     } to @p1::@start_else_0
    // CHECK:   }
    // CHECK: }
    // CHECK: p4hir.state @start_then_0 {
    // CHECK:   p4hir.assign %c1_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p1::@start_post
    // CHECK: }
    // CHECK: p4hir.state @start_else_0 {
    // CHECK:   p4hir.assign %c2_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p1::@start_post
    // CHECK: }
    p4hir.state @start {
      %eq = p4hir.cmp(eq, %arg0, %c2_b8i) : !b8i, !p4hir.bool
      p4hir.if %eq {
        p4hir.assign %c1_b8i, %arg1 : <!b8i>
      } else {
        p4hir.assign %c2_b8i, %arg1 : <!b8i>
      }
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

  // CHECK-LABEL: p4hir.parser @p2
  p4hir.parser @p2(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %c2_b8i = p4hir.const #int2_b8i
    // CHECK: p4hir.state @start_pre_0 {
    // CHECK:   %eq = p4hir.cmp(eq, %arg0, %c2_b8i) : !b8i, !p4hir.bool
    // CHECK:   p4hir.transition_select %eq : !p4hir.bool {
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %{{.*}}
    // CHECK:     } to @p2::@start_then_0
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK:     } to @p2::@start_else_0
    // CHECK:   }
    // CHECK: }
    // CHECK: p4hir.state @start_then_0 {
    // CHECK:   p4hir.transition to @p2::@start_post
    // CHECK: }
    // CHECK: p4hir.state @start_else_0 {
    // CHECK:   p4hir.assign %c2_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p2::@start_post
    // CHECK: }
    p4hir.state @start {
      %eq = p4hir.cmp(eq, %arg0, %c2_b8i) : !b8i, !p4hir.bool
      p4hir.if %eq {
      } else {
        p4hir.assign %c2_b8i, %arg1 : <!b8i>
      }
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

  // CHECK-LABEL: p4hir.parser @p3
  p4hir.parser @p3(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      %eq = p4hir.cmp(eq, %arg0, %c2_b8i) : !b8i, !p4hir.bool
      p4hir.if %eq {
        p4hir.assign %c1_b8i, %arg1 : <!b8i>
      } else {
        p4hir.assign %c2_b8i, %arg1 : <!b8i>
      }
      %val = p4hir.read %arg1 : <!b8i>
      %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
      p4hir.assign %add, %arg1 : <!b8i>
      p4hir.transition to @p3::@accept
    }

    // CHECK: p4hir.state @start_then_0 {
    // CHECK:   p4hir.assign %c1_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p3::@start_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_else_0 {
    // CHECK:   p4hir.assign %c2_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p3::@start_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_post_0 {
    // CHECK:   %val = p4hir.read %arg1 : <!b8i>
    // CHECK:   %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    // CHECK:   p4hir.assign %add, %arg1 : <!b8i>
    // CHECK:   p4hir.parser_accept
    // CHECK: }

    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p3::@start
  }

  // CHECK-LABEL: p4hir.parser @p4
  p4hir.parser @p4(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %c10_b8i = p4hir.const #int10_b8i
    %c3_b8i = p4hir.const #int3_b8i
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i

    // CHECK: p4hir.state @start_pre_0 {
    // CHECK:   %gt = p4hir.cmp(gt, %arg0, %c2_b8i) : !b8i, !p4hir.bool
    // CHECK:   p4hir.transition_select %gt : !p4hir.bool {
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %{{.*}}
    // CHECK:     } to @p4::@start_then_0
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK:     } to @p4::@start_else_0
    // CHECK:   }
    // CHECK: }
    // CHECK: p4hir.state @start_then_0 {
    // CHECK:   p4hir.transition to @p4::@start_then_0_pre_0
    // CHECK: }
    // CHECK: p4hir.state @start_then_0_pre_0 {
    // CHECK:   %eq = p4hir.cmp(eq, %arg0, %c10_b8i) : !b8i, !p4hir.bool
    // CHECK:   p4hir.transition_select %eq : !p4hir.bool {
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %{{.*}}
    // CHECK:     } to @p4::@start_then_0_then_0
    // CHECK:     p4hir.select_case {
    // CHECK:       p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK:     } to @p4::@start_then_0_else_0
    // CHECK:   }
    // CHECK: }
    // CHECK: p4hir.state @start_then_0_then_0 {
    // CHECK:   p4hir.assign %c1_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p4::@start_then_0_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_then_0_else_0 {
    // CHECK:   p4hir.assign %c2_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p4::@start_then_0_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_then_0_post_0 {
    // CHECK:   p4hir.transition to @p4::@start_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_else_0 {
    // CHECK:   p4hir.assign %c3_b8i, %arg1 : <!b8i>
    // CHECK:   p4hir.transition to @p4::@start_post_0
    // CHECK: }
    // CHECK: p4hir.state @start_post_0 {
    // CHECK:   p4hir.parser_accept
    // CHECK: }

    p4hir.state @start {
      %gt = p4hir.cmp(gt, %arg0, %c2_b8i) : !b8i, !p4hir.bool
      p4hir.if %gt {
        %eq = p4hir.cmp(eq, %arg0, %c10_b8i) : !b8i, !p4hir.bool
        p4hir.if %eq {
          p4hir.assign %c1_b8i, %arg1 : <!b8i>
        } else {
          p4hir.assign %c2_b8i, %arg1 : <!b8i>
        }
      } else {
        p4hir.assign %c3_b8i, %arg1 : <!b8i>
      }
      p4hir.transition to @p4::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p4::@start
  }

  // CHECK-LABEL: p4hir.parser @p5
  p4hir.parser @p5(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %c10_b8i = p4hir.const #int10_b8i
    %c2_b8i = p4hir.const #int2_b8i
    %c3_b8i = p4hir.const #int3_b8i
    %c1_b8i = p4hir.const #int1_b8i
    // CHECK: %temp = p4hir.variable ["temp", init] : <!b8i>
    // CHECK: %promoted_local = p4hir.variable ["promoted_local"] : <!b8i>
    // CHECK: p4hir.state @start
    p4hir.state @start {
      %add = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
      %temp = p4hir.variable ["temp", init] : <!b8i>
      p4hir.assign %add, %temp : <!b8i>
      %gt = p4hir.cmp(gt, %arg0, %c2_b8i) : !b8i, !p4hir.bool
      p4hir.if %gt {
        %val_3 = p4hir.read %temp : <!b8i>
        %add_4 = p4hir.binop(add, %val_3, %c1_b8i) : !b8i
        p4hir.assign %add_4, %temp : <!b8i>
        %eq = p4hir.cmp(eq, %arg0, %c10_b8i) : !b8i, !p4hir.bool
        p4hir.if %eq {
          p4hir.assign %c1_b8i, %arg1 : <!b8i>
        } else {
          %val_5 = p4hir.read %temp : <!b8i>
          %add_6 = p4hir.binop(add, %val_5, %c3_b8i) : !b8i
          p4hir.assign %add_6, %temp : <!b8i>
          p4hir.assign %c2_b8i, %arg1 : <!b8i>
        }
      } else {
        p4hir.assign %c3_b8i, %arg1 : <!b8i>
      }
      %val = p4hir.read %temp : <!b8i>
      %add_0 = p4hir.binop(add, %val, %add) : !b8i
      %div = p4hir.binop(div, %add_0, %c2_b8i) : !b8i
      %val_1 = p4hir.read %arg1 : <!b8i>
      %add_2 = p4hir.binop(add, %val_1, %div) : !b8i
      p4hir.assign %add_2, %arg1 : <!b8i>
      p4hir.transition to @p5::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p5::@start
  }
}
