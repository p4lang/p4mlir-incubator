// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!b32i = !p4hir.bit<32>
!int = !p4hir.infint

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
  p4hir.func @blackhole_int(!int)

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

    // CHECK-DAG: %[[cM4_infint:.*]] = p4hir.const #int-4_infint
    // CHECK-DAG: %[[cM255_infint:.*]] = p4hir.const #int-255_infint
    // CHECK-DAG: %[[cM256_infint:.*]] = p4hir.const #int-256_infint
    // CHECK-DAG: %[[c510_infint:.*]] = p4hir.const #int510_infint
    // CHECK-DAG: %[[c0_infint:.*]] = p4hir.const #int0_infint
    // CHECK-DAG: %[[c2_infint:.*]] = p4hir.const #int2_infint
    // CHECK-DAG: %[[c3_infint:.*]] = p4hir.const #int3_infint
    // CHECK-DAG: %[[c116_infint:.*]] = p4hir.const #int116_infint
    // CHECK-DAG: %[[c178_infint:.*]] = p4hir.const #int178_infint
    // CHECK-DAG: %[[c65025_infint:.*]] = p4hir.const #int65025_infint

    %c0_i32i = p4hir.const #int0_i32i
    %c1_i32i = p4hir.const #int1_i32i
    %c5_i32i = p4hir.const #int5_i32i
    %c10_i32i = p4hir.const #int10_i32i
    %c15_i32i = p4hir.const #int15_i32i
    %c50_i32i = p4hir.const #int50_i32i
    %c0_b32i = p4hir.const #int0_b32i
    %cones_b32i = p4hir.const #int-1_b32i

    %c255_int = p4hir.const #p4hir.int<255> : !int
    %c45506_int = p4hir.const #p4hir.int<45506> : !int
    %cMinus1_int = p4hir.const #p4hir.int<-1> : !int
    %cMinus3_int = p4hir.const #p4hir.int<-3> : !int

    // ADD

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res1 = p4hir.binop(add, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res1) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res2 = p4hir.binop(add, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res2) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res3 = p4hir.binop(add, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res3) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res4 = p4hir.binop(add, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res4) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[cM4_infint]])
    %res5 = p4hir.binop(add, %cMinus1_int, %cMinus3_int) : !int
    p4hir.call @blackhole_int(%res5) : (!int) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c510_infint]])
    %res6 = p4hir.binop(add, %c255_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res6) : (!int) -> ()

    // SADD

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res7 = p4hir.binop(sadd, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res7) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res8 = p4hir.binop(sadd, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res8) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res9 = p4hir.binop(sadd, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res9) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res10 = p4hir.binop(sadd, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res10) : (!b32i) -> ()

    // SUB

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res11 = p4hir.binop(sub, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res11) : (!i32i) -> ()

    // CHECK: %[[minus:.*]] = p4hir.unary(minus, %[[signed]])
    // CHECK: p4hir.call @blackhole_i32i (%[[minus]])
    %res12 = p4hir.binop(sub, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res12) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res13 = p4hir.binop(sub, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res13) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c2_infint]])
    %res14 = p4hir.binop(sub, %cMinus1_int, %cMinus3_int) : !int
    p4hir.call @blackhole_int(%res14) : (!int) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[cM256_infint]])
    %res15 = p4hir.binop(sub, %cMinus1_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res15) : (!int) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c0_infint]])
    %res16 = p4hir.binop(sub, %c255_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res16) : (!int) -> ()

    // SSUB

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res17 = p4hir.binop(ssub, %arg_i32i, %c0_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res17) : (!i32i) -> ()

    // CHECK: %[[minus:.*]] = p4hir.unary(minus, %[[signed]])
    // CHECK: p4hir.call @blackhole_i32i (%[[minus]])
    %res18 = p4hir.binop(ssub, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res18) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res19 = p4hir.binop(sub, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res19) : (!b32i) -> ()

    // MUL

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res20 = p4hir.binop(mul, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res20) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res21 = p4hir.binop(mul, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res21) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c50_i32i]])
    %res22 = p4hir.binop(mul, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res22) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res23 = p4hir.binop(mul, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res23) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c3_infint]])
    %res24 = p4hir.binop(mul, %cMinus1_int, %cMinus3_int) : !int
    p4hir.call @blackhole_int(%res24) : (!int) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[cM255_infint]])
    %res25 = p4hir.binop(mul, %cMinus1_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res25) : (!int) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c65025_infint]])
    %res26 = p4hir.binop(mul, %c255_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res26) : (!int) -> ()

    // DIV

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res27 = p4hir.binop(div, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res27) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res28 = p4hir.binop(div, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res28) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res29 = p4hir.binop(div, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res29) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c178_infint]])
    %res30 = p4hir.binop(div, %c45506_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res30) : (!int) -> ()

    // MOD

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res31 = p4hir.binop(mod, %arg_i32i, %c1_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res31) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res32 = p4hir.binop(mod, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res32) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res33 = p4hir.binop(div, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res33) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_int (%[[c116_infint]])
    %res34 = p4hir.binop(mod, %c45506_int, %c255_int) : !int
    p4hir.call @blackhole_int(%res34) : (!int) -> ()

    // OR

    // CHECK: p4hir.call @blackhole_i32i (%[[signed]])
    %res35 = p4hir.binop(or, %c0_i32i, %arg_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res35) : (!i32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[cones_b32i]])
    %res36 = p4hir.binop(or, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res36) : (!b32i) -> ()

    // AND

    // CHECK: p4hir.call @blackhole_b32i (%[[c0_b32i]])
    %res37 = p4hir.binop(and, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res37) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res38 = p4hir.binop(and, %arg_b32i, %cones_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res38) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c0_i32i]])
    %res39 = p4hir.binop(and, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res39) : (!i32i) -> ()

    // XOR

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res40 = p4hir.binop(xor, %arg_b32i, %c0_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res40) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_b32i (%[[unsigned]])
    %res41 = p4hir.binop(xor, %c0_b32i, %arg_b32i) : !b32i
    p4hir.call @blackhole_b32i(%res41) : (!b32i) -> ()

    // CHECK: p4hir.call @blackhole_i32i (%[[c15_i32i]])
    %res42 = p4hir.binop(xor, %c5_i32i, %c10_i32i) : !i32i
    p4hir.call @blackhole_i32i(%res42) : (!i32i) -> ()

    p4hir.call @blackhole_i32i(%c0_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c1_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c5_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_i32i(%c10_i32i) : (!i32i) -> ()
    p4hir.call @blackhole_b32i(%c0_b32i) : (!b32i) -> ()
    p4hir.call @blackhole_b32i(%cones_b32i) : (!b32i) -> ()
    p4hir.return
  }
}
