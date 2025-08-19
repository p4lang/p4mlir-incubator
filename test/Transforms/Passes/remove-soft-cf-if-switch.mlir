// RUN: p4mlir-opt --p4hir-remove-soft-cf %s | FileCheck %s

!b8i = !p4hir.bit<8>
#int-56_b8i = #p4hir.int<200> : !b8i
#int0_b8i = #p4hir.int<0> : !b8i
#int100_b8i = #p4hir.int<100> : !b8i
#int10_b8i = #p4hir.int<10> : !b8i
#int11_b8i = #p4hir.int<11> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int20_b8i = #p4hir.int<20> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i

module {
  // Test simple transformation to if/else.
  // No return guards should be emitted.
  // void f1(inout bit<8> a) {
  //   a += 1;
  //   if (a == 100) return;
  //   a += 2;
  // }
  // CHECK-LABEL p4hir.func @f1
  p4hir.func @f1(%arg0: !p4hir.ref<!b8i>) {
    %c100_b8i = p4hir.const #int100_b8i
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>
    %val_0 = p4hir.read %arg0 : <!b8i>
    %eq = p4hir.cmp(eq, %val_0 : !b8i, %c100_b8i : !b8i)
    p4hir.if %eq {
      p4hir.soft_return
    }
    %val_1 = p4hir.read %arg0 : <!b8i>
    %add_2 = p4hir.binop(add, %val_1, %c2_b8i) : !b8i
    p4hir.assign %add_2, %arg0 : <!b8i>

    // CHECK: p4hir.if %{{.*}} {
    //          ...
    // CHECK: } else {
    // CHECK:   p4hir.assign %{{.*}}, %arg0 : <!b8i>
    // CHECK: }
    // CHECK-NOT: p4hir.read %return_guard
    // CHECK: p4hir.return

    p4hir.return
  }

  // Also test functions within overload set.
  p4hir.overload_set @f2 {
    // Test multiple returns.
    // No return guards should be emitted.
    // void f2(inout bit<8> a) {
    //   a += 1;
    //   if (a == 100) return;
    //   if (a == 200) return;
    //   a += 2;
    // }
    // CHECK-LABEL p4hir.func @f2_0
    p4hir.func @f2_0(%arg0: !p4hir.ref<!b8i>) {
      %c-56_b8i = p4hir.const #int-56_b8i
      %c100_b8i = p4hir.const #int100_b8i
      %c2_b8i = p4hir.const #int2_b8i
      %c1_b8i = p4hir.const #int1_b8i
      %val = p4hir.read %arg0 : <!b8i>
      %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
      p4hir.assign %add, %arg0 : <!b8i>
      %val_0 = p4hir.read %arg0 : <!b8i>
      %eq = p4hir.cmp(eq, %val_0 : !b8i, %c100_b8i : !b8i)
      p4hir.if %eq {
        p4hir.soft_return
      }
      %val_1 = p4hir.read %arg0 : <!b8i>
      %eq_2 = p4hir.cmp(eq, %val_1 : !b8i, %c-56_b8i : !b8i)
      p4hir.if %eq_2 {
        p4hir.soft_return
      }
      %val_3 = p4hir.read %arg0 : <!b8i>
      %add_4 = p4hir.binop(add, %val_3, %c2_b8i) : !b8i
      p4hir.assign %add_4, %arg0 : <!b8i>

      // CHECK: p4hir.if %{{.*}} {
      //          ...
      // CHECK: } else {
      // CHECK:   p4hir.if %{{.*}} {
      //            ...
      // CHECK:   } else {
      // CHECK:     p4hir.assign %{{.*}}, %arg0 : <!b8i>
      // CHECK:   }
      // CHECK: }
      // CHECK-NOT: p4hir.read %return_guard
      // CHECK: p4hir.return

      p4hir.return
    }

    // Simple test for return value.
    // No return guards should be emitted.
    // bit<8> f2(inout bit<8> a) {
    //   if (a > 100) { return 1; }
    //   return 2;
    // }
    // CHECK-LABEL p4hir.func @f2_1
    p4hir.func @f2_1(%arg0: !p4hir.ref<!b8i>) -> !b8i {
      // CHECK: %return_value = p4hir.variable ["return_value", init] : <!b8i>
      %c2_b8i = p4hir.const #int2_b8i
      %c1_b8i = p4hir.const #int1_b8i
      %c100_b8i = p4hir.const #int100_b8i
      %val = p4hir.read %arg0 : <!b8i>
      %gt = p4hir.cmp(gt, %val : !b8i, %c100_b8i : !b8i)
      p4hir.if %gt {
        p4hir.soft_return %c1_b8i : !b8i
      }
      p4hir.soft_return %c2_b8i : !b8i

      // CHECK: p4hir.if %{{.*}} {
      // CHECK:   p4hir.assign %c1_b8i, %return_value : <!b8i>
      // CHECK: } else {
      // CHECK:   p4hir.assign %c2_b8i, %return_value : <!b8i>
      // CHECK: }
      // CHECK-NOT: p4hir.read %return_guard
      // CHECK: %[[RES:.*]] = p4hir.read %return_value : <!b8i>
      // CHECK: p4hir.return %[[RES]] : !b8i

      p4hir.return
    }
  }

  // Test mutliple executions points in statement.
  // One guard needed after the if statement.
  // bit<8> f3(inout bit<8> a) {
  //   if (a < 100) {
  //   } else if (a == 200) {
  //     return 0;
  //   }
  //   return a;
  // }
  // CHECK-LABEL p4hir.func @f3
  p4hir.func @f3(%arg0: !p4hir.ref<!b8i>) -> !b8i {
    %c-56_b8i = p4hir.const #int-56_b8i
    %c0_b8i = p4hir.const #int0_b8i
    %c100_b8i = p4hir.const #int100_b8i
    %val = p4hir.read %arg0 : <!b8i>
    %lt = p4hir.cmp(lt, %val : !b8i, %c100_b8i : !b8i)
    p4hir.if %lt {
    } else {
      %val_1 = p4hir.read %arg0 : <!b8i>
      %eq = p4hir.cmp(eq, %val_1 : !b8i, %c-56_b8i : !b8i)
      p4hir.if %eq {
        p4hir.soft_return %c0_b8i : !b8i
      }
    }
    %val_0 = p4hir.read %arg0 : <!b8i>
    p4hir.soft_return %val_0 : !b8i

    // CHECK: %[[GUARD:.*]] = p4hir.read %return_guard : <!p4hir.bool>
    // CHECK: p4hir.if %[[GUARD]] {
    // CHECK:   %[[A_VAR:.*]] = p4hir.read %arg0 : <!b8i>
    // CHECK:   p4hir.assign %[[A_VAR]], %return_value : <!b8i>
    // CHECK: }
    // CHECK: %[[RES:.*]] = p4hir.read %return_value : <!b8i>
    // CHECK: p4hir.return %[[RES]] : !b8i

    p4hir.return
  }

  // Test switch statement.
  // No return guards should be emitted.
  // bit<8> f4(inout bit<8> a) {
  //   switch (a) {
  //     1: { return 10; }
  //     2: {  }
  //     default: { return 11; }
  //   }
  //   return a;
  // }
  // CHECK-LABEL p4hir.func @f4
  p4hir.func @f4(%arg0: !p4hir.ref<!b8i>) -> !b8i {
    %c11_b8i = p4hir.const #int11_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %val = p4hir.read %arg0 : <!b8i>
    p4hir.switch (%val : !b8i) {
      p4hir.case(equal, [#int1_b8i]) {
        p4hir.soft_return %c10_b8i : !b8i
        p4hir.yield
      }
      p4hir.case(equal, [#int2_b8i]) {
        p4hir.yield
      }
      p4hir.case(default, []) {
        p4hir.soft_return %c11_b8i : !b8i
        p4hir.yield
      }
      p4hir.yield
    }
    %val_0 = p4hir.read %arg0 : <!b8i>
    p4hir.soft_return %val_0 : !b8i

    // CHECK: p4hir.switch (%{{.*}}) {
    // CHECK:   p4hir.case(equal, [#int1_b8i]) {
    // CHECK:     p4hir.assign %c10_b8i, %return_value : <!b8i>
    // CHECK:   }
    // CHECK:   p4hir.case(equal, [#int2_b8i]) {
    // CHECK:     %[[A_VAR:.*]] = p4hir.read %arg0 : <!b8i>
    // CHECK:     p4hir.assign %[[A_VAR]], %return_value : <!b8i>
    // CHECK:   }
    // CHECK:   p4hir.case(default, []) {
    // CHECK:     p4hir.assign %c11_b8i, %return_value : <!b8i>
    // CHECK:   }
    // CHECK: }
    // CHECK-NOT: p4hir.read %return_guard
    // CHECK: %[[RES:.*]] = p4hir.read %return_value : <!b8i>
    // CHECK: p4hir.return %[[RES]] : !b8i

    p4hir.return
  }

  // Test switch statement with mutliple executions points.
  // One guard needed after the switch statement.
  // bit<8> f5(inout bit<8> a) {
  //   switch (a) {
  //     1: { return 10; }
  //     2: { a += 1; }
  //     3: { return 20; }
  //     default: {  }
  //   }
  //   return a;
  // }
  // CHECK-LABEL p4hir.func @f5
  p4hir.func @f5(%arg0: !p4hir.ref<!b8i>) -> !b8i {
    %c20_b8i = p4hir.const #int20_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %val = p4hir.read %arg0 : <!b8i>
    p4hir.switch (%val : !b8i) {
      p4hir.case(equal, [#int1_b8i]) {
        p4hir.soft_return %c10_b8i : !b8i
        p4hir.yield
      }
      p4hir.case(equal, [#int2_b8i]) {
        %val_1 = p4hir.read %arg0 : <!b8i>
        %add = p4hir.binop(add, %val_1, %c1_b8i) : !b8i
        p4hir.assign %add, %arg0 : <!b8i>
        p4hir.yield
      }
      p4hir.case(equal, [#int3_b8i]) {
        p4hir.soft_return %c20_b8i : !b8i
        p4hir.yield
      }
      p4hir.case(default, []) {
        p4hir.yield
      }
      p4hir.yield
    }
    %val_0 = p4hir.read %arg0 : <!b8i>
    p4hir.soft_return %val_0 : !b8i

    // CHECK: %[[GUARD:.*]] = p4hir.read %return_guard : <!p4hir.bool>
    // CHECK: p4hir.if %[[GUARD]] {
    // CHECK:   %[[A_VAR:.*]] = p4hir.read %arg0 : <!b8i>
    // CHECK:   p4hir.assign %[[A_VAR]], %return_value : <!b8i>
    // CHECK: }
    // CHECK: %[[RES:.*]] = p4hir.read %return_value : <!b8i>
    // CHECK: p4hir.return %[[RES]] : !b8i

    p4hir.return
  }

  // Identify and eliminate unreachable code.
  // bit<8> f6(inout bit<8> a) {
  //   if (a != 100) {
  //     return 1;
  //   } else {
  //     return 2;
  //   }
  //   // unreachable.
  //   a = 10;
  //   return 20;
  // }
  // CHECK-LABEL p4hir.func @f6
  p4hir.func @f6(%arg0: !p4hir.ref<!b8i>) -> !b8i {
    %c20_b8i = p4hir.const #int20_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c100_b8i = p4hir.const #int100_b8i
    %val = p4hir.read %arg0 : <!b8i>
    %ne = p4hir.cmp(ne, %val : !b8i, %c100_b8i : !b8i)
    // CHECK: p4hir.if
    // CHECK-NOT: %c10_b8i
    // CHECK-NOT: %c20_b8i
    // CHECK: p4hir.return
    p4hir.if %ne {
      p4hir.soft_return %c1_b8i : !b8i
    } else {
      p4hir.soft_return %c2_b8i : !b8i
    }
    p4hir.assign %c10_b8i, %arg0 : <!b8i>
    p4hir.soft_return %c20_b8i : !b8i
    p4hir.return
  }

  // Test more complex example.
  // No return guards should be emitted.
  // bit<8> f7(inout bit<8> a) {
  //   switch (a) {
  //     1: { return 10; }
  //     2: {
  //       if (a > 100) {
  //         return a;
  //       }
  //     }
  //     default: { return 11; }
  //   }
  //   if (a < 10) {
  //     return a;
  //   }
  //   a = 1;
  //   return a;
  // }
  // CHECK-LABEL p4hir.func @f7
  p4hir.func @f7(%arg0: !p4hir.ref<!b8i>) -> !b8i {
    %c100_b8i = p4hir.const #int100_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c11_b8i = p4hir.const #int11_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %val = p4hir.read %arg0 : <!b8i>
    // CHECK-NOT: p4hir.read %return_guard
    p4hir.switch (%val : !b8i) {
      p4hir.case(equal, [#int1_b8i]) {
        p4hir.soft_return %c10_b8i : !b8i
        p4hir.yield
      }
      p4hir.case(equal, [#int2_b8i]) {
        %val_2 = p4hir.read %arg0 : <!b8i>
        %gt = p4hir.cmp(gt, %val_2 : !b8i, %c100_b8i : !b8i)
        p4hir.if %gt {
          %val_3 = p4hir.read %arg0 : <!b8i>
          p4hir.soft_return %val_3 : !b8i
        } else {
        }
        p4hir.yield
      }
      p4hir.case(default, []) {
        p4hir.soft_return %c11_b8i : !b8i
        p4hir.yield
      }
      p4hir.yield
    }
    %val_0 = p4hir.read %arg0 : <!b8i>
    %lt = p4hir.cmp(lt, %val_0 : !b8i, %c10_b8i : !b8i)
    p4hir.if %lt {
      %val_2 = p4hir.read %arg0 : <!b8i>
      p4hir.soft_return %val_2 : !b8i
    }
    p4hir.assign %c1_b8i, %arg0 : <!b8i>
    %val_1 = p4hir.read %arg0 : <!b8i>
    p4hir.soft_return %val_1 : !b8i
    p4hir.return
  }
}
