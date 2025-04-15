// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!int  = !p4hir.infint

#int0_b8i = #p4hir.int<0> : !b8i
#int9_b8i = #p4hir.int<9> : !b8i

// CHECK: module
module  {
  // CHECK: %[[c0:.*]] = p4hir.const #int0_b8i
  %c0 = p4hir.const #int0_b8i

  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_i8i(!i8i)
  p4hir.func @blackhole_int(!int)

  // CHECK-LABEL: p4hir.func @test_shift_zero(%arg0: !b8i)
  p4hir.func @test_shift_zero(%arg_b8 : !b8i) {
    // CHECK: p4hir.call @blackhole_b8i (%arg0) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8, %c0 : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_shift_width(%arg0: !b8i)
  p4hir.func @test_shift_width(%arg_b8 : !b8i) {
    %c9 = p4hir.const #int9_b8i

    // CHECK: p4hir.call @blackhole_b8i (%[[c0]]) : (!b8i) -> ()
    %shl0 = p4hir.shl(%arg_b8, %c9 : !b8i) : !b8i
    p4hir.call @blackhole_b8i(%shl0) : (!b8i) -> ()

    p4hir.return

  }
}

