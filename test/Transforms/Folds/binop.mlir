// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!b32i = !p4hir.bit<32>

#int0_i32i = #p4hir.int<0> : !i32i
#int1_i32i = #p4hir.int<1> : !i32i
#int5_i32i = #p4hir.int<5> : !i32i
#int10_i32i = #p4hir.int<10> : !i32i
#int15_i32i = #p4hir.int<15> : !i32i
#int50_i32i = #p4hir.int<50> : !i32i
#int0_b32i = #p4hir.int<0> : !b32i
#int-1_b32i = #p4hir.int<4294967295> : !b32i  // 0xFFFFFFFF for 32-bit

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_i32i(!i32i)
  p4hir.func @blackhole_b32i(!b32i)

  // CHECK: p4hir.func @test(%[[signed:.*]]: !i32i, %[[unsigned:.*]]: !b32i)
  p4hir.func @test(%arg_i32i : !i32i, %arg_b32i : !b32i) {
    // CHECK-DAG: %[[c0_i32i:.*]] = p4hir.const #int0_i32i
    // CHECK-DAG: %[[c1_i32i:.*]] = p4hir.const #int1_i32i
    // CHECK-DAG: %[[c5_i32i:.*]] = p4hir.const #int5_i32i
    // CHECK-DAG: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
    // CHECK-DAG: %[[c15_i32i:.*]] = p4hir.const #int15_i32i
    // CHECK-DAG: %[[c50_i32i:.*]] = p4hir.const #int50_i32i
    // CHECK-DAG: %[[c0_b32i:.*]] = p4hir.const #int0_b32i
    // CHECK-DAG: %[[cones_b32i:.*]] = p4hir.const #int-1_b32i
    %c0_i32i = p4hir.const #int0_i32i
    %c1_i32i = p4hir.const #int1_i32i
    %c5_i32i = p4hir.const #int5_i32i
    %c10_i32i = p4hir.const #int10_i32i
    %c15_i32i = p4hir.const #int15_i32i
    %c50_i32i = p4hir.const #int50_i32i
    %c0_b32i = p4hir.const #int0_b32i
    %cones_b32i = p4hir.const #int-1_b32i

    // MUL

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res1 = p4hir.binop(mul, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res1) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res2 = p4hir.binop(mul, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res2) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c50_i32i]])
    %res3 = p4hir.binop(mul, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res3) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res4 = p4hir.binop(mul, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res4) : (!b32i) -> ()

    // DIV

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res5 = p4hir.binop(div, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res5) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res6 = p4hir.binop(div, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res6) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res7 = p4hir.binop(div, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res7) : (!b32i) -> ()

    // MOD

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res8 = p4hir.binop(mod, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res8) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res9 = p4hir.binop(mod, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res9) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res10 = p4hir.binop(div, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res10) : (!b32i) -> ()

    // ADD

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res11 = p4hir.binop(add, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res11) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res12 = p4hir.binop(add, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res12) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res13 = p4hir.binop(add, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res13) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res14 = p4hir.binop(add, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res14) : (!b32i) -> ()

    // SADD

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res15 = p4hir.binop(sadd, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res15) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res16 = p4hir.binop(sadd, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res16) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res17 = p4hir.binop(sadd, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res17) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res18 = p4hir.binop(sadd, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res18) : (!b32i) -> ()

    // SUB

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res19 = p4hir.binop(sub, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res19) : (!i32i) -> ()

    // CHECK: %[[minus:.*]] = p4hir.unary(minus, %[[signed]])
    // CHECK: p4hir.call @blackhole_i32i (%[[minus]])
    %res20 = p4hir.binop(sub, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res20) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res21 = p4hir.binop(sub, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res21) : (!b32i) -> ()

    // SSUB

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res22 = p4hir.binop(ssub, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res22) : (!i32i) -> ()

    // CHECK: %[[minus:.*]] = p4hir.unary(minus, %[[signed]])
    // CHECK: p4hir.call @blackhole_i32i (%[[minus]])
    %res23 = p4hir.binop(ssub, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res23) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res24 = p4hir.binop(sub, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res24) : (!b32i) -> ()

    // OR

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res25 = p4hir.binop(or, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res25) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[cones_b32i]])
    %res26 = p4hir.binop(or, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res26) : (!b32i) -> ()

    // AND

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res27 = p4hir.binop(and, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res27) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res28 = p4hir.binop(and, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res28) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res29 = p4hir.binop(and, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res29) : (!i32i) -> ()

    // XOR

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res30 = p4hir.binop(xor, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res30) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res31 = p4hir.binop(xor, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res31) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res32 = p4hir.binop(xor, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res32) : (!i32i) -> ()

    p4hir.call @blackhole_i32i(%c0_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c1_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c5_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c10_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_b32i(%c0_b32i) : (!b32i) -> ()
    p4hir.call @blackhole_b32i(%cones_b32i) : (!b32i) -> ()
    p4hir.return
  }
}

