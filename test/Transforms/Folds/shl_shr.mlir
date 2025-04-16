// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!int = !p4hir.infint

#int0_b8i = #p4hir.int<0> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i
#int5_b8i = #p4hir.int<5> : !b8i
#int8_b8i = #p4hir.int<8> : !b8i
#int9_b8i = #p4hir.int<9> : !b8i
#int40_b8i = #p4hir.int<40> : !b8i

#int0_i8i = #p4hir.int<0> : !i8i
#int-1_i8i = #p4hir.int<-1> : !i8i
#int-5_i8i = #p4hir.int<-5> : !i8i
#int-40_i8i = #p4hir.int<-40> : !i8i


// CHECK: module
module  {
  // CHECK-DAG: %[[c0_b8i:.*]] = p4hir.const #int0_b8i
  // CHECK-DAG: %[[c3_b8i:.*]] = p4hir.const #int3_b8i
  // CHECK-DAG: %[[c5_b8i:.*]] = p4hir.const #int5_b8i
  // CHECK-DAG: %[[c8_b8i:.*]] = p4hir.const #int8_b8i
  // CHECK-DAG: %[[c9_b8i:.*]] = p4hir.const #int9_b8i
  // CHECK-DAG: %[[c40_b8i:.*]] = p4hir.const #int40_b8i
  // CHECK-DAG: %[[c0_i8i:.*]] = p4hir.const #int0_i8i
  // CHECK-DAG: %[[cminus1_i8i:.*]] = p4hir.const #int-1_i8i
  // CHECK-DAG: %[[cminus5_i8i:.*]] = p4hir.const #int-5_i8i
  // CHECK-DAG: %[[cminus40_i8i:.*]] = p4hir.const #int-40_i8i
  %c0_b8i = p4hir.const #int0_b8i
  %c3_b8i = p4hir.const #int3_b8i
  %c5_b8i = p4hir.const #int5_b8i
  %c8_b8i = p4hir.const #int8_b8i
  %c9_b8i = p4hir.const #int9_b8i
  %c40_b8i = p4hir.const #int40_b8i
  %c0_i8i = p4hir.const #int0_i8i
  %c-1_i8i = p4hir.const #int-1_i8i
  %c-5_i8i = p4hir.const #int-5_i8i
  %c-40_i8i = p4hir.const #int-40_i8i

  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_i8i(!i8i)
  p4hir.func @blackhole_int(!int)

  // CHECK-LABEL: p4hir.func @test_shift_zero_identity(%arg0: !b8i, %arg1: !i8i)
  p4hir.func @test_shift_zero_identity(%arg_b8i : !b8i, %arg_i8i : !i8i) {
    // ===== ShlOp =====

    // CHECK: p4hir.call @blackhole_b8i (%arg0) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_b8i (%[[c8_b8i]]) : (!b8i) -> ()
    %shl1 = p4hir.shl(%c8_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl1) : (!b8i) -> ()

    // ===== ShrOp =====

    // CHECK: p4hir.call @blackhole_b8i (%arg0) : (!b8i) -> ()
    %shr0 = p4hir.shr(%arg_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shr0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%arg1) : (!i8i) -> ()
    %shr1 = p4hir.shr(%arg_i8i, %c0_b8i : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%shr1) : (!i8i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_shift_ge_width(%arg0: !b8i, %arg1: !i8i)
  p4hir.func @test_shift_ge_width(%arg_b8i : !b8i, %arg_i8i : !i8i) {
    // ===== ShlOp =====

    // CHECK: p4hir.call @blackhole_b8i (%[[c0_b8i]]) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8i, %c9_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[c0_i8i]]) : (!i8i) -> ()
    %shl1 = p4hir.shl(%arg_i8i, %c8_b8i : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%shl1) : (!i8i) -> ()

    // ===== ShrOp =====

    // CHECK: p4hir.call @blackhole_b8i (%[[c0_b8i]]) : (!b8i) -> ()
    %shr0 = p4hir.shr(%arg_b8i, %c8_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shr0) : (!b8i) -> ()

    // CHECK: %[[shr:.*]] = p4hir.shr(%arg1, %[[c9_b8i]] : !b8i) : !i8i
    // CHECK: p4hir.call @blackhole_i8i (%[[shr]]) : (!i8i) -> ()
    %shr1 = p4hir.shr(%arg_i8i, %c9_b8i : !b8i) : !i8i // arg >> 9 = no fold (signed var)
    p4hir.call @blackhole_i8i(%shr1) : (!i8i) -> ()

    // 0b11111111 = -1
    // CHECK: p4hir.call @blackhole_i8i (%[[cminus1_i8i]]) : (!i8i) -> ()
    %shr2 = p4hir.shr(%c-5_i8i, %c9_b8i : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%shr2) : (!i8i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_shift_const()
  p4hir.func @test_shift_const() {
    // ===== ShlOp =====

    // CHECK: p4hir.call @blackhole_b8i (%[[c40_b8i]]) : (!b8i) -> ()
    %shl0 = p4hir.shl(%c5_b8i, %c3_b8i : !b8i) : !b8i // 5 << 3 = 40
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[cminus40_i8i]]) : (!i8i) -> ()
    %shl1 = p4hir.shl(%c-5_i8i, %c3_b8i : !b8i) : !i8i // -5 << 3 = -40
    p4hir.call @blackhole_i8i(%shl1) : (!i8i) -> ()

    // ===== ShrOp =====

    // CHECK: p4hir.call @blackhole_b8i (%[[c0_b8i]]) : (!b8i) -> ()
    %shr0 = p4hir.shr(%c5_b8i, %c3_b8i : !b8i) : !b8i // 5 >> 3 (logical) = 0
    p4hir.call @blackhole_b8i(%shr0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[cminus1_i8i]]) : (!i8i) -> ()
    %shr1 = p4hir.shr(%c-5_i8i, %c3_b8i : !b8i) : !i8i // -5 >> 3 (arith) = -1
    p4hir.call @blackhole_i8i(%shr1) : (!i8i) -> ()

    p4hir.return
  }

  // Ensure these constants stay in the module
  p4hir.call @blackhole_b8i(%c3_b8i) : (!b8i) -> ()
  p4hir.call @blackhole_b8i(%c5_b8i) : (!b8i) -> ()
  p4hir.call @blackhole_i8i(%c-5_i8i) : (!i8i) -> ()
  p4hir.call @blackhole_i8i(%c-40_i8i) : (!i8i) -> ()
}

