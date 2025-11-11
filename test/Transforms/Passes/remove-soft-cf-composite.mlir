// RUN: p4mlir-opt --pass-pipeline='builtin.module(p4hir.func(p4hir-remove-soft-cf, canonicalize))' %s | FileCheck %s

// Test combination of all soft control flow statements.

!b8i = !p4hir.bit<8>
#int0_b8i = #p4hir.int<0> : !b8i
#int10_b8i = #p4hir.int<10> : !b8i
#int15_b8i = #p4hir.int<15> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int20_b8i = #p4hir.int<20> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i

module {
  // Original function:
  // bit<8> f1(inout bit<8> a, inout bit<8> b) {
  //   for (bit<8> i = 0; i < a; i += 1) {
  //     bit<8> j = i + 1;
  //     if (i % 2 == 0) {
  //       for (; j < b; j *= 3) {
  //         switch (j) {
  //           10: { continue; }
  //           15: { break; }
  //           20: { return a; }
  //           default: { j += 1; }
  //         }
  //         b += j;
  //       }
  //     } else if (i == b) {
  //       continue;
  //     }
  // 
  //     b += 3;
  //   }
  // 
  //   return a + b;
  // }

  // Transformed function:
  // bit<8> f1(inout bit<8> a, inout bit<8> b) {
  //   bool returnGuard = true;
  //   bit<8> returnValue;
  // 
  //   bool continueGuard1 = true;
  //   for (bit<8> i = 0; (i < a && returnGuard); i += 1) {
  //     continueGuard1 = true;
  //     bit<8> j = i + 1;
  //     if (i % 2 == 0) {
  //       bool breakGuard2 = true;
  //       for (; (j < b && breakGuard2); j *= 3) {
  //         switch (j) {
  //           10: {}
  //           15: {
  //             breakGuard2 = true;
  //           }
  //           20: {
  //             returnGuard = true;
  //             breakGuard2 = true;
  //             returnValue = a;
  //           }
  //           default: {
  //             j += 1;
  //             b += j;
  //           }
  //         }
  //       }
  //     } else if (i == b) {
  //       continueGuard1 = false;
  //     }
  // 
  //     if (continueGuard1) {
  //       b += 3;
  //     }
  //   }
  //   
  //   if (returnGuard) {
  //     returnValue = a + b;
  //   }
  //   return returnValue;
  // }

  // CHECK-LABEL: p4hir.func @f1
  p4hir.func @f1(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "a"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "b"}) -> !b8i {
    %c3_b8i = p4hir.const #int3_b8i
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.scope {
      %i = p4hir.variable ["i", init] : <!b8i>
      p4hir.assign %c0_b8i, %i : <!b8i>
      p4hir.for : cond {
        %val_1 = p4hir.read %i : <!b8i>
        %val_2 = p4hir.read %arg0 : <!b8i>
        %lt = p4hir.cmp(lt, %val_1 : !b8i, %val_2 : !b8i)
        p4hir.condition %lt
      } body {
        %val_1 = p4hir.read %i : <!b8i>
        %add_2 = p4hir.binop(add, %val_1, %c1_b8i) : !b8i
        %j = p4hir.variable ["j", init] : <!b8i>
        p4hir.assign %add_2, %j : <!b8i>
        %val_3 = p4hir.read %i : <!b8i>
        %mod = p4hir.binop(mod, %val_3, %c2_b8i) : !b8i
        %eq = p4hir.cmp(eq, %mod : !b8i, %c0_b8i : !b8i)
        p4hir.if %eq {
          p4hir.for : cond {
            %val_6 = p4hir.read %j : <!b8i>
            %val_7 = p4hir.read %arg1 : <!b8i>
            %lt = p4hir.cmp(lt, %val_6 : !b8i, %val_7 : !b8i)
            p4hir.condition %lt
          } body {
            %val_6 = p4hir.read %j : <!b8i>
            p4hir.switch (%val_6 : !b8i) {
              p4hir.case(equal, [#int10_b8i]) {
                p4hir.soft_continue
                p4hir.yield
              }
              p4hir.case(equal, [#int15_b8i]) {
                p4hir.soft_break
                p4hir.yield
              }
              p4hir.case(equal, [#int20_b8i]) {
                %val_10 = p4hir.read %arg0 : <!b8i>
                p4hir.soft_return %val_10 : !b8i
                p4hir.yield
              }
              p4hir.case(default, []) {
                %val_10 = p4hir.read %j : <!b8i>
                %add_11 = p4hir.binop(add, %val_10, %c1_b8i) : !b8i
                p4hir.assign %add_11, %j : <!b8i>
                p4hir.yield
              }
              p4hir.yield
            }
            %val_7 = p4hir.read %arg1 : <!b8i>
            %val_8 = p4hir.read %j : <!b8i>
            %add_9 = p4hir.binop(add, %val_7, %val_8) : !b8i
            p4hir.assign %add_9, %arg1 : <!b8i>
            p4hir.yield
          } updates {
            %val_6 = p4hir.read %j : <!b8i>
            %mul = p4hir.binop(mul, %val_6, %c3_b8i) : !b8i
            p4hir.assign %mul, %j : <!b8i>
            p4hir.yield
          }
        } else {
          %val_6 = p4hir.read %i : <!b8i>
          %val_7 = p4hir.read %arg1 : <!b8i>
          %eq_8 = p4hir.cmp(eq, %val_6 : !b8i, %val_7 : !b8i)
          p4hir.if %eq_8 {
            p4hir.soft_continue
          }
        }
        %val_4 = p4hir.read %arg1 : <!b8i>
        %add_5 = p4hir.binop(add, %val_4, %c3_b8i) : !b8i
        p4hir.assign %add_5, %arg1 : <!b8i>
        p4hir.yield
      } updates {
        %val_1 = p4hir.read %i : <!b8i>
        %add_2 = p4hir.binop(add, %val_1, %c1_b8i) : !b8i
        p4hir.assign %add_2, %i : <!b8i>
        p4hir.yield
      }
    }
    %val = p4hir.read %arg0 : <!b8i>
    %val_0 = p4hir.read %arg1 : <!b8i>
    %add = p4hir.binop(add, %val, %val_0) : !b8i
    p4hir.soft_return %add : !b8i

    // CHECK: %[[RETURN_GUARD:.*]] = p4hir.variable ["return_guard", init] : <!p4hir.bool>
    // CHECK: %[[RETURN_VALUE:.*]] = p4hir.variable ["return_value", init] : <!b8i>
    //        ...
    // CHECK: %[[CONTINUE_GUARD_1:.*]] = p4hir.variable ["loop_continue_guard", init] : <!p4hir.bool>
    // CHECK: p4hir.assign %true, %[[CONTINUE_GUARD_1]] : <!p4hir.bool>
    // CHECK: p4hir.for : cond {
    // CHECK:   %[[GUARD_VAL_1:.*]] = p4hir.read %[[RETURN_GUARD]] : <!p4hir.bool>
    // CHECK:   %[[NEW_COND_1:.*]] = p4hir.ternary(%[[GUARD_VAL_1]], true {
    //            ...
    // CHECK:     %[[ORIG_COND_1:.*]] = p4hir.cmp(lt, %{{.*}} : !b8i, %{{.*}} : !b8i)
    // CHECK:     p4hir.yield %[[ORIG_COND_1]] : !p4hir.bool
    // CHECK:   }, false {
    // CHECK:     p4hir.yield %false : !p4hir.bool
    // CHECK:   }) : !p4hir.bool
    // CHECK:   p4hir.condition %[[NEW_COND_1]]
    // CHECK: } body {
    // CHECK:   p4hir.assign %true, %[[CONTINUE_GUARD_1]] : <!p4hir.bool>
    //          ...
    // CHECK:   p4hir.if %eq {
    // CHECK:     %[[BREAK_GUARD_2:.*]] = p4hir.variable ["loop_break_guard", init] : <!p4hir.bool>
    // CHECK:     p4hir.assign %true, %[[BREAK_GUARD_2]] : <!p4hir.bool>
    // CHECK:     p4hir.for : cond {
    // CHECK:       %[[GUARD_VAL_2:.*]] = p4hir.read %[[BREAK_GUARD_2]] : <!p4hir.bool>
    // CHECK:       %[[NEW_COND_2:.*]] = p4hir.ternary(%[[GUARD_VAL_2]], true {
    //                ...
    // CHECK:         %[[ORIG_COND_2:.*]] = p4hir.cmp(lt, %{{.*}} : !b8i, %{{.*}} : !b8i)
    // CHECK:         p4hir.yield %[[ORIG_COND_2]] : !p4hir.bool
    // CHECK:       }, false {
    // CHECK:         p4hir.yield %false : !p4hir.bool
    // CHECK:       }) : !p4hir.bool
    // CHECK:       p4hir.condition %[[NEW_COND_2]]
    // CHECK:     } body {
    // CHECK:       %[[SWITCH_VAL:.*]] = p4hir.read %j : <!b8i>
    // CHECK:       p4hir.switch (%[[SWITCH_VAL]] : !b8i) {
    // CHECK:         p4hir.case(equal, [#int10_b8i]) {
    // CHECK:           p4hir.yield
    // CHECK:         }
    // CHECK:         p4hir.case(equal, [#int15_b8i]) {
    // CHECK:           p4hir.assign %false, %[[BREAK_GUARD_2]] : <!p4hir.bool>
    // CHECK:           p4hir.yield
    // CHECK:         }
    // CHECK:         p4hir.case(equal, [#int20_b8i]) {
    // CHECK:           %[[ARG0_VAL:.*]] = p4hir.read %arg0 : <!b8i>
    // CHECK:           p4hir.assign %false, %[[RETURN_GUARD]] : <!p4hir.bool>
    // CHECK:           p4hir.assign %false, %[[BREAK_GUARD_2]] : <!p4hir.bool>
    // CHECK:           p4hir.assign %[[ARG0_VAL]], %[[RETURN_VALUE]] : <!b8i>
    // CHECK:           p4hir.yield
    // CHECK:         }
    // CHECK:         p4hir.case(default, []) {
    // CHECK:           %val_5 = p4hir.read %j : <!b8i>
    // CHECK:           %add_6 = p4hir.binop(add, %val_5, %c1_b8i) : !b8i
    // CHECK:           p4hir.assign %add_6, %j : <!b8i>
    // CHECK:           %val_7 = p4hir.read %arg1 : <!b8i>
    // CHECK:           %val_8 = p4hir.read %j : <!b8i>
    // CHECK:           %add_9 = p4hir.binop(add, %val_7, %val_8) : !b8i
    // CHECK:           p4hir.assign %add_9, %arg1 : <!b8i>
    // CHECK:           p4hir.yield
    // CHECK:         }
    // CHECK:         p4hir.yield
    // CHECK:       }
    // CHECK:       p4hir.yield
    // CHECK:     } updates {
    // CHECK:       %[[GUARD_VAL_2_UPDATES:.*]] = p4hir.read %[[BREAK_GUARD_2]] : <!p4hir.bool>
    // CHECK:       p4hir.if %[[GUARD_VAL_2_UPDATES]] {
    // CHECK:         %val_5 = p4hir.read %j : <!b8i>
    // CHECK:         %mul = p4hir.binop(mul, %val_5, %c3_b8i) : !b8i
    // CHECK:         p4hir.assign %mul, %j : <!b8i>
    // CHECK:       }
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:   } else {
    // CHECK:     %val_4 = p4hir.read %i : <!b8i>
    // CHECK:     %val_5 = p4hir.read %arg1 : <!b8i>
    // CHECK:     %eq_6 = p4hir.cmp(eq, %val_4 : !b8i, %val_5 : !b8i)
    // CHECK:     p4hir.if %eq_6 {
    // CHECK:       p4hir.assign %false, %[[CONTINUE_GUARD_1]] : <!p4hir.bool>
    // CHECK:     } else {
    // CHECK:     }
    // CHECK:   }
    // CHECK:   %val_3 = p4hir.read %[[CONTINUE_GUARD_1]] : <!p4hir.bool>
    // CHECK:   p4hir.if %val_3 {
    // CHECK:     %val_4 = p4hir.read %arg1 : <!b8i>
    // CHECK:     %add_5 = p4hir.binop(add, %val_4, %c3_b8i) : !b8i
    // CHECK:     p4hir.assign %add_5, %arg1 : <!b8i>
    // CHECK:   }
    // CHECK:   p4hir.yield
    // CHECK: } updates {
    // CHECK:   %[[GUARD_VAL_UPDATES:.*]] = p4hir.read %[[RETURN_GUARD]] : <!p4hir.bool>
    // CHECK:   p4hir.if %[[GUARD_VAL_UPDATES]] {
    // CHECK:     %val_2 = p4hir.read %i : <!b8i>
    // CHECK:     %add = p4hir.binop(add, %val_2, %c1_b8i) : !b8i
    // CHECK:     p4hir.assign %add, %i : <!b8i>
    // CHECK:   }
    // CHECK:   p4hir.yield
    // CHECK: }
    // CHECK: %[[GUARD_VAL:.*]] = p4hir.read %[[RETURN_GUARD]] : <!p4hir.bool>
    // CHECK: p4hir.if %[[GUARD_VAL]] {
    // CHECK:   %val_1 = p4hir.read %arg0 : <!b8i>
    // CHECK:   %val_2 = p4hir.read %arg1 : <!b8i>
    // CHECK:   %add = p4hir.binop(add, %val_1, %val_2) : !b8i
    // CHECK:   p4hir.assign %false, %[[RETURN_GUARD]] : <!p4hir.bool>
    // CHECK:   p4hir.assign %add, %[[RETURN_VALUE]] : <!b8i>
    // CHECK: }
    // CHECK: %val_0 = p4hir.read %[[RETURN_VALUE]] : <!b8i>
    // CHECK: p4hir.return %val_0 : !b8i

    p4hir.return
  }
}
