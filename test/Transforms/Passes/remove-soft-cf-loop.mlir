// RUN: p4mlir-opt --p4hir-remove-soft-cf --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#int0_b8i = #p4hir.int<0> : !b8i
#int100_b8i = #p4hir.int<100> : !b8i
#int10_b8i = #p4hir.int<10> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int8_b8i = #p4hir.int<8> : !b8i
#int9_b8i = #p4hir.int<9> : !b8i

module {
  // Check addition of return guard in the loop's condition.
  // void f1(inout bit<8> a) {
  //   for (bit<8> i = 0; i < a; i += 1) {
  //     if (i == 10) { return; }
  //   }
  //   a += 1;
  // }
  // CHECK-LABEL: p4hir.func @f1
  p4hir.func @f1(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}) {
    %c10_b8i = p4hir.const #int10_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.scope {
      %i = p4hir.variable ["i", init] : <!b8i>
      p4hir.assign %c0_b8i, %i : <!b8i>
      p4hir.for : cond {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg0 : <!b8i>
        %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
        p4hir.condition %lt
      } body {
        %val_0 = p4hir.read %i : <!b8i>
        %eq = p4hir.cmp(eq, %val_0 : !b8i, %c10_b8i : !b8i)
        p4hir.if %eq {
          p4hir.soft_return
        }
        p4hir.yield
      } updates {
        %val_0 = p4hir.read %i : <!b8i>
        %add_1 = p4hir.binop(add, %val_0, %c1_b8i) : !b8i
        p4hir.assign %add_1, %i : <!b8i>
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>

    // CHECK: p4hir.for : cond {
    //            ...
    // CHECK:     %[[ORIG_COND:.*]] = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
    // CHECK:     %[[NEW_COND:.*]] = p4hir.ternary(%[[ORIG_COND]], true {
    // CHECK:       %[[GUARD_COND:.*]] = p4hir.read %return_guard : <!p4hir.bool>
    // CHECK:       p4hir.yield %[[GUARD_COND]] : !p4hir.bool
    // CHECK:     }, false {
    // CHECK:       p4hir.yield %false : !p4hir.bool
    // CHECK:     }) : !p4hir.bool
    // CHECK:     p4hir.condition %[[NEW_COND]]
    // CHECK:   } body {
    //            ...
    // CHECK:     p4hir.if %{{.*}} {
    // CHECK:       p4hir.assign %false, %return_guard : <!p4hir.bool>
    // CHECK:     }
    // CHECK:   }
    //            ...
    // CHECK: }

    // CHECK: %[[GUARD:.*]] = p4hir.read %return_guard : <!p4hir.bool>
    // CHECK: p4hir.if %[[GUARD]] {
    // CHECK:   %{{.*}} = p4hir.read %arg0 : <!b8i>
    // CHECK:   %{{.*}} = p4hir.binop(add, %{{.*}}, %c1_b8i) : !b8i
    // CHECK:   p4hir.assign %{{.*}}, %arg0 : <!b8i>
    // CHECK: }

    p4hir.return
  }

  // void f2(inout bit<8> a) {
  //   for (bit<8> i in 1..a) {
  //     if (i == 10) { return; }
  //   }
  //   a += 1;
  // }
  // CHECK-LABEL: p4hir.func @f2
  p4hir.func @f2(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}) {
    %c10_b8i = p4hir.const #int10_b8i
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.scope {
      %val_0 = p4hir.read %arg0 : <!b8i>
      %range = p4hir.range(%c1_b8i, %val_0) : !p4hir.set<!b8i>

      // CHECK:      p4hir.foreach %arg1 : !b8i in %range : !p4hir.set<!b8i> {
      // CHECK-NEXT:   %[[GUARD:.*]] = p4hir.read %return_guard : <!p4hir.bool>
      // CHECK-NEXT:   p4hir.if %[[GUARD]] {
      //                 ...
      // CHECK:        }
      // CHECK:        p4hir.yield
      // CHECK:      }

      p4hir.foreach %arg1 : !b8i in %range : !p4hir.set<!b8i> {
        %eq = p4hir.cmp(eq, %arg1 : !b8i, %c10_b8i : !b8i)
        p4hir.if %eq {
          p4hir.soft_return
        }
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>
    p4hir.return
  }

  // Check returning from double-nested loop.
  // void f3a(inout bit<8> a) {
  //   for (bit<8> i = 0; i < a; i += 1) {
  //     for (bit<8> j = 0; j < a; j += 1) {
  //       if (i + j == 100) { return; }
  //     }
  //   }
  //   a += 1;
  // }
  // CHECK-LABEL: p4hir.func @f3a
  p4hir.func @f3a(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}) {
    %c100_b8i = p4hir.const #int100_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.scope {
      %i = p4hir.variable ["i", init] : <!b8i>
      p4hir.assign %c0_b8i, %i : <!b8i>
      p4hir.for : cond {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg0 : <!b8i>
        %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
        p4hir.condition %lt
      } body {
        p4hir.scope {
          %j = p4hir.variable ["j", init] : <!b8i>
          p4hir.assign %c0_b8i, %j : <!b8i>
          p4hir.for : cond {
            %val_0 = p4hir.read %j : <!b8i>
            %val_1 = p4hir.read %arg0 : <!b8i>
            %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
            p4hir.condition %lt
          } body {
            %val_0 = p4hir.read %i : <!b8i>
            %val_1 = p4hir.read %j : <!b8i>
            %add_2 = p4hir.binop(add, %val_0, %val_1) : !b8i
            %eq = p4hir.cmp(eq, %add_2 : !b8i, %c100_b8i : !b8i)
            p4hir.if %eq {
              p4hir.soft_return
            }
            p4hir.yield
          } updates {
            %val_0 = p4hir.read %j : <!b8i>
            %add_1 = p4hir.binop(add, %val_0, %c1_b8i) : !b8i
            p4hir.assign %add_1, %j : <!b8i>
            p4hir.yield
          }
        }
        p4hir.yield
      } updates {
        %val_0 = p4hir.read %i : <!b8i>
        %add_1 = p4hir.binop(add, %val_0, %c1_b8i) : !b8i
        p4hir.assign %add_1, %i : <!b8i>
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>

    // CHECK: p4hir.for : cond {
    // CHECK:   %[[NEW_COND_1:.*]] = p4hir.ternary
    // CHECK:     p4hir.read %return_guard
    //          ...
    // CHECK:   p4hir.condition %[[NEW_COND_1]]
    // CHECK: } body {
    // CHECK:   p4hir.for : cond {
    // CHECK:     %[[NEW_COND_2:.*]] = p4hir.ternary
    // CHECK:       p4hir.read %return_guard
    //            ...
    // CHECK:     p4hir.condition %[[NEW_COND_2]]
    // CHECK:   } body {
    //            ...
    // CHECK:     p4hir.if %{{.*}} {
    // CHECK:       p4hir.assign %false, %return_guard : <!p4hir.bool>
    // CHECK:     }
    // CHECK:   }
    // CHECK: }
    // CHECK: %[[GUARD:.*]] = p4hir.read %return_guard : <!p4hir.bool>
    // CHECK: p4hir.if %[[GUARD]] {
    // CHECK:   %{{.*}} = p4hir.read %arg0 : <!b8i>
    // CHECK:   %{{.*}} = p4hir.binop(add, %{{.*}}, %c1_b8i) : !b8i
    // CHECK:   p4hir.assign %{{.*}}, %arg0 : <!b8i>
    // CHECK: }

    p4hir.return
  }

  // Check breaking from inner loop in double-nested loop.
  // void f3b(inout bit<8> a) {
  //   for (bit<8> i = 0; i < a; i += 1) {
  //     for (bit<8> j = 0; j < a; j += 1) {
  //       if (i + j == 100) { break; }
  //     }
  //   }
  //   a += 1;
  // }
  // CHECK-LABEL: p4hir.func @f3b
  p4hir.func @f3b(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}) {
    %c100_b8i = p4hir.const #int100_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.scope {
      %i = p4hir.variable ["i", init] : <!b8i>
      p4hir.assign %c0_b8i, %i : <!b8i>
      p4hir.for : cond {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg0 : <!b8i>
        %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
        p4hir.condition %lt
      } body {
        p4hir.scope {
          %j = p4hir.variable ["j", init] : <!b8i>
          p4hir.assign %c0_b8i, %j : <!b8i>
          p4hir.for : cond {
            %val_0 = p4hir.read %j : <!b8i>
            %val_1 = p4hir.read %arg0 : <!b8i>
            %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
            p4hir.condition %lt
          } body {
            %val_0 = p4hir.read %i : <!b8i>
            %val_1 = p4hir.read %j : <!b8i>
            %add_2 = p4hir.binop(add, %val_0, %val_1) : !b8i
            %eq = p4hir.cmp(eq, %add_2 : !b8i, %c100_b8i : !b8i)
            p4hir.if %eq {
              p4hir.soft_break
            }
            p4hir.yield
          } updates {
            %val_0 = p4hir.read %j : <!b8i>
            %add_1 = p4hir.binop(add, %val_0, %c1_b8i) : !b8i
            p4hir.assign %add_1, %j : <!b8i>
            p4hir.yield
          }
        }
        p4hir.yield
      } updates {
        %val_0 = p4hir.read %i : <!b8i>
        %add_1 = p4hir.binop(add, %val_0, %c1_b8i) : !b8i
        p4hir.assign %add_1, %i : <!b8i>
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>

    // CHECK: p4hir.for : cond {
    // CHECK:   %[[OLD_COND:.*]] = p4hir.cmp(lt, %{{.*}} : !b8i, %{{.*}} : !b8i)
    // CHECK:   p4hir.condition %[[OLD_COND]]
    // CHECK: } body {
    // CHECK:   p4hir.for : cond {
    // CHECK:     %[[NEW_COND:.*]] = p4hir.ternary
    // CHECK:       p4hir.read %loop_break_guard
    //            ...
    // CHECK:     p4hir.condition %[[NEW_COND]]
    // CHECK:   } body {
    //            ...
    // CHECK:     p4hir.if %{{.*}} {
    // CHECK:       p4hir.assign %false, %loop_break_guard : <!p4hir.bool>
    // CHECK:     }
    // CHECK:   }
    // CHECK: }
    //        No return guard is needed.
    // CHECK: %{{.*}} = p4hir.read %arg0 : <!b8i>
    // CHECK: %{{.*}} = p4hir.binop(add, %{{.*}}, %c1_b8i) : !b8i
    // CHECK: p4hir.assign %{{.*}}, %arg0 : <!b8i>
    // CHECK-NEXT: p4hir.return

    p4hir.return
  }

  // CHECK break and return in a loop.
  // void f4(inout bit<8> a, inout bit<8> b) {
  //   for (bit<8> i = 0; i < a; i += b) {
  //     if (i % b == 8) {
  //       break;
  //     } else if (i % 10 == 9) {
  //       return;
  //     }
  //     
  //     a += b; // no return guard here.
  //   }
  //   a += 1;
  // }
  // CHECK-LABEL: p4hir.func @f4
  p4hir.func @f4(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "b"}) {
    %c9_b8i = p4hir.const #int9_b8i
    %c8_b8i = p4hir.const #int8_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.scope {
      %i = p4hir.variable ["i", init] : <!b8i>
      p4hir.assign %c0_b8i, %i : <!b8i>
      p4hir.for : cond {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg0 : <!b8i>
        %lt = p4hir.cmp(lt, %val_0 : !b8i, %val_1 : !b8i)
        p4hir.condition %lt
      } body {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg1 : <!b8i>
        %mod = p4hir.binop(mod, %val_0, %val_1) : !b8i
        %eq = p4hir.cmp(eq, %mod : !b8i, %c8_b8i : !b8i)
        p4hir.if %eq {
          p4hir.soft_break
        } else {
          %val_5 = p4hir.read %i : <!b8i>
          %mod_6 = p4hir.binop(mod, %val_5, %c10_b8i) : !b8i
          %eq_7 = p4hir.cmp(eq, %mod_6 : !b8i, %c9_b8i : !b8i)
          p4hir.if %eq_7 {
            p4hir.soft_return
          }
        }
        %val_2 = p4hir.read %arg0 : <!b8i>
        %val_3 = p4hir.read %arg1 : <!b8i>
        %add_4 = p4hir.binop(add, %val_2, %val_3) : !b8i
        p4hir.assign %add_4, %arg0 : <!b8i>
        p4hir.yield
      } updates {
        %val_0 = p4hir.read %i : <!b8i>
        %val_1 = p4hir.read %arg1 : <!b8i>
        %add_2 = p4hir.binop(add, %val_0, %val_1) : !b8i
        p4hir.assign %add_2, %i : <!b8i>
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
    p4hir.assign %add, %arg0 : <!b8i>

    // CHECK: body {
    // CHECK:   %[[EQ_1:.*]] = p4hir.cmp(eq, %{{.*}} : !b8i, %c8_b8i : !b8i)
    // CHECK:   p4hir.if %[[EQ_1]] {
    // CHECK:     p4hir.assign %{{.*}}, %loop_break_guard : <!p4hir.bool>
    // CHECK:   } else {
    // CHECK:     %[[EQ_2:.*]] = p4hir.cmp(eq, %{{.*}} : !b8i, %c9_b8i : !b8i)
    // CHECK:     p4hir.if %[[EQ_2]] {
    // CHECK:       p4hir.assign %{{.*}}, %loop_break_guard : <!p4hir.bool>
    // CHECK:       p4hir.assign %{{.*}}, %return_guard : <!p4hir.bool>
    // CHECK:     } else {
    // CHECK:       %[[ARG_1:.*]] = p4hir.read %arg0 : <!b8i>
    // CHECK:       %[[ARG_2:.*]] = p4hir.read %arg1 : <!b8i>
    // CHECK:       %add = p4hir.binop(add, %[[ARG_1]], %[[ARG_2]]) : !b8i
    // CHECK:       p4hir.assign %add, %arg0 : <!b8i>
    // CHECK:     }
    // CHECK:   }
    // CHECK: }

    p4hir.return
  }
}
