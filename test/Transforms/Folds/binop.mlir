// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!b32i = !p4hir.bit<32>

#int0_i32i = #p4hir.int<0> : !i32i
#int1_i32i = #p4hir.int<1> : !i32i
#int0_b32i = #p4hir.int<0> : !b32i
#int-1_b32i = #p4hir.int<4294967295> : !b32i  // 0xFFFFFFFF for 32-bit

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_i32i(!i32i)
  p4hir.func @blackhole_b32i(!b32i)

  // CHECK-DAG: %[[c0_i32i:.*]] = p4hir.const #int0_i32i
  // CHECK-DAG: %[[c1_i32i:.*]] = p4hir.const #int1_i32i
  // CHECK-DAG: %[[c0_b32i:.*]] = p4hir.const #int0_b32i
  // CHECK-DAG: %[[cones_b32i:.*]] = p4hir.const #int-1_b32i
  %c0_i32i = p4hir.const #int0_i32i
  %c1_i32i = p4hir.const #int1_i32i
  %c0_b32i = p4hir.const #int0_b32i
  %cones_b32i = p4hir.const #int-1_b32i

  // CHECK: p4hir.func @test_binop(%[[signed:.*]]: !i32i, %[[unsigned:.*]]: !b32i)
  p4hir.func @test_binop(%arg_i32i : !i32i, %arg_b32i : !b32i) {
    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res1 = p4hir.binop(add, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res1) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res2 = p4hir.binop(add, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res2) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res3 = p4hir.binop(sub, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res3) : (!i32i) -> ()

    // CHECK: %[[minus:.*]] = p4hir.unary(minus, %[[signed]])
    // CHECK: p4hir.call @blackhole_i32i (%[[minus]])
    %res4 = p4hir.binop(sub, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res4) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res5 = p4hir.binop(mul, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res5) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res6 = p4hir.binop(mul, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res6) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res7 = p4hir.binop(div, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res7) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res8 = p4hir.binop(div, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res8) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res9 = p4hir.binop(mod, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res9) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res10 = p4hir.binop(mod, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res10) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res11 = p4hir.binop(or, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res11) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[cones_b32i]])
    %res12 = p4hir.binop(or, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res12) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res13 = p4hir.binop(and, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res13) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res14 = p4hir.binop(and, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res14) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res15 = p4hir.binop(xor, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res15) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res16 = p4hir.binop(xor, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res16) : (!b32i) -> ()

    p4hir.return
  }

  p4hir.call @blackhole_i32i(%c0_i32i) : (!i32i) -> ()
  p4hir.call @blackhole_i32i(%c1_i32i) : (!i32i) -> ()
  p4hir.call @blackhole_b32i(%c0_b32i) : (!b32i) -> ()
  p4hir.call @blackhole_b32i(%cones_b32i) : (!b32i) -> ()
}

