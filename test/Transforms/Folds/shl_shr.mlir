// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!infint = !p4hir.infint

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

#int0_infint = #p4hir.int<0> : !infint
#int3_infint = #p4hir.int<3> : !infint
#int5_infint = #p4hir.int<5> : !infint
#int40_infint = #p4hir.int<40> : !infint
#int62_infint = #p4hir.int<62> : !infint
#int500_infint = #p4hir.int<500> : !infint
#int128000_infint = #p4hir.int<128000> : !infint
#int-1_infint = #p4hir.int<-1> : !infint
#int-5_infint = #p4hir.int<-5> : !infint
#int-40_infint = #p4hir.int<-40> : !infint
#int-63_infint = #p4hir.int<-63> : !infint
#int-500_infint = #p4hir.int<-500> : !infint

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
  // CHECK-DAG: %[[c0_infint:.*]] = p4hir.const #int0_infint
  // CHECK-DAG: %[[c5_infint:.*]] = p4hir.const #int5_infint
  // CHECK-DAG: %[[c40_infint:.*]] = p4hir.const #int40_infint
  // CHECK-DAG: %[[c62_infint:.*]] = p4hir.const #int62_infint
  // CHECK-DAG: %[[c500_infint:.*]] = p4hir.const #int500_infint
  // CHECK-DAG: %[[c128000_infint:.*]] = p4hir.const #int128000_infint
  // CHECK-DAG: %[[cminus1_infint:.*]] = p4hir.const #int-1_infint
  // CHECK-DAG: %[[cminus5_infint:.*]] = p4hir.const #int-5_infint
  // CHECK-DAG: %[[cminus40_infint:.*]] = p4hir.const #int-40_infint
  // CHECK-DAG: %[[cminus63_infint:.*]] = p4hir.const #int-63_infint
  // CHECK-DAG: %[[cminus500_infint:.*]] = p4hir.const #int-500_infint
  %c0_b8i = p4hir.const #int0_b8i
  %c3_b8i = p4hir.const #int3_b8i
  %c5_b8i = p4hir.const #int5_b8i
  %c8_b8i = p4hir.const #int8_b8i
  %c9_b8i = p4hir.const #int9_b8i
  %c40_b8i = p4hir.const #int40_b8i
  %c0_i8i = p4hir.const #int0_i8i
  %c-5_i8i = p4hir.const #int-5_i8i
  %c-40_i8i = p4hir.const #int-40_i8i
  %c0_infint = p4hir.const #int0_infint
  %c5_infint = p4hir.const #int5_infint
  %c40_infint = p4hir.const #int40_infint
  %c62_infint = p4hir.const #int62_infint
  %c500_infint = p4hir.const #int500_infint
  %c128000 = p4hir.const #int128000_infint
  %c-1_infint = p4hir.const #int-1_infint
  %c-5_infint = p4hir.const #int-5_infint
  %c-40_infint = p4hir.const #int-40_infint
  %c-63_infint = p4hir.const #int-63_infint
  %c-500_infint = p4hir.const #int-500_infint

  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_i8i(!i8i)
  p4hir.func @blackhole_infint(!infint)

  // CHECK-LABEL: p4hir.func @test_shift_zero_identity(%arg0: !b8i, %arg1: !i8i, %arg2: !infint)
  p4hir.func @test_shift_zero_identity(%arg_b8i : !b8i, %arg_i8i : !i8i, %arg_infint : !infint) {
    // CHECK: p4hir.call @blackhole_b8i (%arg0) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_b8i (%[[c8_b8i]]) : (!b8i) -> ()
    %shl1 = p4hir.shl(%c8_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl1) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_infint (%arg2) : (!infint) -> ()
    %shl2 = p4hir.shl(%arg_infint, %c0_b8i : !b8i) : !infint
    p4hir.call @blackhole_infint(%shl2) : (!infint) -> ()


    // CHECK: p4hir.call @blackhole_b8i (%arg0) : (!b8i) -> ()
    %shr0 = p4hir.shr(%arg_b8i, %c0_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shr0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%arg1) : (!i8i) -> ()
    %shr1 = p4hir.shr(%arg_i8i, %c0_b8i : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%shr1) : (!i8i) -> ()
   
    // CHECK: p4hir.call @blackhole_infint (%arg2) : (!infint) -> ()
    %shr2 = p4hir.shr(%arg_infint, %c0_b8i : !b8i) : !infint
    p4hir.call @blackhole_infint(%shr2) : (!infint) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_shift_ge_width(%arg0: !b8i, %arg1: !i8i)
  p4hir.func @test_shift_ge_width(%arg_b8i : !b8i, %arg_i8i : !i8i) {
    // CHECK: p4hir.call @blackhole_b8i (%[[c0_b8i]]) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8i, %c9_b8i : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[c0_i8i]]) : (!i8i) -> ()
    %shl1 = p4hir.shl(%arg_i8i, %c8_b8i : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%shl1) : (!i8i) -> ()


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
    // CHECK: p4hir.call @blackhole_b8i (%[[c40_b8i]]) : (!b8i) -> ()
    %shl0 = p4hir.shl(%c5_b8i, %c3_b8i : !b8i) : !b8i // 5 << 3 = 40
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[cminus40_i8i]]) : (!i8i) -> ()
    %shl1 = p4hir.shl(%c-5_i8i, %c3_b8i : !b8i) : !i8i // -5 << 3 = -40
    p4hir.call @blackhole_i8i(%shl1) : (!i8i) -> ()

    // CHECK: p4hir.call @blackhole_b8i (%[[c0_b8i]]) : (!b8i) -> ()
    %shr0 = p4hir.shr(%c5_b8i, %c3_b8i : !b8i) : !b8i // 5 >> 3 (logical) = 0
    p4hir.call @blackhole_b8i(%shr0) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_i8i (%[[cminus1_i8i]]) : (!i8i) -> ()
    %shr1 = p4hir.shr(%c-5_i8i, %c3_b8i : !b8i) : !i8i // -5 >> 3 (arith) = -1
    p4hir.call @blackhole_i8i(%shr1) : (!i8i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_shift_infint()
  p4hir.func @test_shift_infint() {
    // CHECK: p4hir.call @blackhole_infint (%[[c40_infint]]) : (!infint) -> ()
    %shl0 = p4hir.shl(%c5_infint, %c3_b8i : !b8i) : !infint // 5 << 3 = 40
    p4hir.call @blackhole_infint(%shl0) : (!infint) -> ()

    // CHECK: p4hir.call @blackhole_infint (%[[cminus40_infint]]) : (!infint) -> ()
    %shl1 = p4hir.shl(%c-5_infint, %c3_b8i : !b8i) : !infint // -5 << 3 = -40
    p4hir.call @blackhole_infint(%shl1) : (!infint) -> ()

    // CHECK: p4hir.call @blackhole_infint (%[[c128000_infint]]) : (!infint) -> ()
    %shl2 = p4hir.shl(%c500_infint, %c8_b8i : !b8i) : !infint // 500 << 8 = 128000
    p4hir.call @blackhole_infint(%shl2) : (!infint) -> ()


    // CHECK: p4hir.call @blackhole_infint (%[[c0_infint]]) : (!infint) -> ()
    %shr0 = p4hir.shr(%c5_infint, %c3_b8i : !b8i) : !infint // 5 >> 3 = 0
    p4hir.call @blackhole_infint(%shr0) : (!infint) -> ()

    // CHECK: p4hir.call @blackhole_infint (%[[cminus1_infint]]) : (!infint) -> ()
    %shr1 = p4hir.shr(%c-5_infint, %c3_b8i : !b8i) : !infint // -5 >> 3 = -1
    p4hir.call @blackhole_infint(%shr1) : (!infint) -> ()

    // CHECK: p4hir.call @blackhole_infint (%[[c62_infint]]) : (!infint) -> ()
    %shr2 = p4hir.shr(%c500_infint, %c3_b8i : !b8i) : !infint // 500 >> 3 = 62
    p4hir.call @blackhole_infint(%shr2) : (!infint) -> ()

    // CHECK: p4hir.call @blackhole_infint (%[[cminus63_infint]]) : (!infint) -> ()
    %shr3 = p4hir.shr(%c-500_infint, %c3_b8i : !b8i) : !infint // -500 >> 3 = -63
    p4hir.call @blackhole_infint(%shr3) : (!infint) -> ()

    p4hir.return
  }

  // Make sure these constants don't get DCE'd
  p4hir.call @blackhole_b8i(%c3_b8i) : (!b8i) -> ()
  p4hir.call @blackhole_b8i(%c5_b8i) : (!b8i) -> ()
  p4hir.call @blackhole_i8i(%c-5_i8i) : (!i8i) -> ()
  p4hir.call @blackhole_i8i(%c-40_i8i) : (!i8i) -> ()
  p4hir.call @blackhole_infint(%c0_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c5_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c500_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c-1_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c-5_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c-40_infint) : (!infint) -> ()
  p4hir.call @blackhole_infint(%c-500_infint) : (!infint) -> ()
}

