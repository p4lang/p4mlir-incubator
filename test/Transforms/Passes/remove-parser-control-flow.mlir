// RUN: p4mlir-opt --p4hir-remove-parser-control-flow %s | FileCheck %s

!b32i = !p4hir.bit<32>

#int0_b32i = #p4hir.int<0> : !b32i
#int10_b32i = #p4hir.int<10> : !b32i
#int12_b32i = #p4hir.int<12> : !b32i
#int15_b32i = #p4hir.int<15> : !b32i
#int16_b32i = #p4hir.int<16> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int32_b32i = #p4hir.int<32> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int63_b32i = #p4hir.int<63> : !b32i
#int64_b32i = #p4hir.int<64> : !b32i
#int92_b32i = #p4hir.int<92> : !b32i

// remove control flow from a state with the equivalent of:
// switch (b) {
//     16:
//     32: {
//         if (a == 0) { b = 1; } else { b = 2; }
//     }
//     63:
//     64: {
//       switch (a) {
//         1: { b = 10; }
//         2: { b = 12; }
//         default: { b = 15; }
//       }
//     }
//     92:
//     default: { b = 3; }
// }

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.parser @p1
  p4hir.parser @p1(%arg0: !p4hir.ref<!b32i>, %arg1: !p4hir.ref<!b32i>)() {
    %c0_b32i = p4hir.const #int0_b32i
    %c3_b32i = p4hir.const #int3_b32i
    %c15_b32i = p4hir.const #int15_b32i
    %c12_b32i = p4hir.const #int12_b32i
    %c10_b32i = p4hir.const #int10_b32i
    %c2_b32i = p4hir.const #int2_b32i
    %c1_b32i = p4hir.const #int1_b32i

    p4hir.state @start {
      %val = p4hir.read %arg1 : <!b32i>
      p4hir.switch (%val : !b32i) {
        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
          %val_0 = p4hir.read %arg0 : <!b32i>
          %eq = p4hir.cmp(eq, %val_0, %c0_b32i) : !b32i, !p4hir.bool
          p4hir.if %eq {
            p4hir.assign %c1_b32i, %arg1 : <!b32i>
          } else {
            p4hir.assign %c2_b32i, %arg1 : <!b32i>
          }
          p4hir.yield
        }
        p4hir.case(anyof, [#int63_b32i, #int64_b32i]) {
          %val_0 = p4hir.read %arg0 : <!b32i>
          p4hir.switch (%val_0 : !b32i) {
            p4hir.case(equal, [#int1_b32i]) {
              p4hir.assign %c10_b32i, %arg1 : <!b32i>
              p4hir.yield
            }
            p4hir.case(equal, [#int2_b32i]) {
              p4hir.assign %c12_b32i, %arg1 : <!b32i>
              p4hir.yield
            }
            p4hir.case(default, []) {
              p4hir.assign %c15_b32i, %arg1 : <!b32i>
              p4hir.yield
            }
            p4hir.yield
          }
          p4hir.yield
        }
        p4hir.case(default, [#int92_b32i]) {
          p4hir.assign %c3_b32i, %arg1 : <!b32i>
          p4hir.yield
        }
        p4hir.yield
      }
      p4hir.transition to @p1::@accept
    }

    // Check the structure of generated states.

    // CHECK: p4hir.state @start

    // Lowering of outter switch.
    // CHECK: p4hir.state @start_pre
    // CHECK: %val = p4hir.read %arg1 : <!b32i>
    // CHECK: p4hir.transition_select %val : !b32i

    // First case of outter switch.
    // CHECK: p4hir.state @start_case0_0
    
    // Lowering of if statement.
    // CHECK: p4hir.state @start_case0_0_pre_0
    // CHECK: p4hir.state @start_case0_0_then_0
    // CHECK: p4hir.transition to @p1::@start_case0_0_post_0
    // CHECK: p4hir.state @start_case0_0_else_0
    // CHECK: p4hir.transition to @p1::@start_case0_0_post_0
    // CHECK: p4hir.state @start_case0_0_post_0
    // CHECK: p4hir.transition to @p1::@start_post_0

    // Lowering of inner switch.
    // CHECK: p4hir.state @start_case1_0_pre_0
    // CHECK: %val = p4hir.read %arg0 : <!b32i>
    // CHECK: p4hir.transition_select %val : !b32i

    // Cases of inner switch.
    // CHECK: p4hir.state @start_case1_0_case0_0
    // CHECK: p4hir.state @start_case1_0_case1_0
    // CHECK: p4hir.state @start_case1_0_case2_0
    
    // CHECK: p4hir.state @start_case1_0_post_0

    // Last case of outter switch
    // CHECK: p4hir.state @start_case2_0

    // Done.
    // CHECK: p4hir.state @start_post_0
    // CHECK: p4hir.transition to @p1::@accept

    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p1::@start
  }
}
