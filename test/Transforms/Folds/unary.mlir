// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
!b  = !p4hir.bool

#int5_b8i = #p4hir.int<5> : !b8i
#int-5_b8i = #p4hir.int<-5> : !b8i
#int-6_b8i = #p4hir.int<-6> : !b8i
#true = #p4hir.bool<true> : !b
#false = #p4hir.bool<false> : !b

// CHECK-LABEL: module
module {
  // CHECK-DAG: %[[true:.*]] = p4hir.const #true
  // CHECK-DAG: %[[false:.*]] = p4hir.const #false
  // CHECK-DAG: %[[cminus6_b8i:.*]] = p4hir.const #int-6_b8i
  // CHECK-DAG: %[[cminus5_b8i:.*]] = p4hir.const #int-5_b8i
  // CHECK-DAG: %[[c5:.*]] = p4hir.const #int5_b8i
  %true = p4hir.const #true
  %false = p4hir.const #false
  %c-6_b8i = p4hir.const #int-6_b8i
  %c-5_b8i = p4hir.const #int-5_b8i
  %c5_b8i = p4hir.const #int5_b8i

  p4hir.func @blackhole(!b8i)
  p4hir.func @blackhole_bool(!b)

  // CHECK-LABEL: p4hir.func @test_unary_const
  p4hir.func @test_unary_const() {
    // CHECK: p4hir.call @blackhole (%[[cminus5_b8i]]) : (!b8i) -> ()
    %r1 = p4hir.unary(minus, %c5_b8i) : !b8i
    p4hir.call @blackhole(%r1) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole (%[[c5]]) : (!b8i) -> ()
    %r2 = p4hir.unary(plus, %c5_b8i) : !b8i
    p4hir.call @blackhole(%r2) : (!b8i) -> ()

    // ~5 = -6
    // CHECK: p4hir.call @blackhole (%[[cminus6_b8i]]) : (!b8i) -> ()
    %r3 = p4hir.unary(cmpl, %c5_b8i) : !b8i
    p4hir.call @blackhole(%r3) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_bool (%[[false]]) : (!p4hir.bool) -> ()
    %r4 = p4hir.unary(not, %true) : !b
    p4hir.call @blackhole_bool(%r4) : (!b) -> ()

    // CHECK: p4hir.call @blackhole_bool (%[[true]]) : (!p4hir.bool) -> ()
    %r5 = p4hir.unary(not, %false) : !b
    p4hir.call @blackhole_bool(%r5) : (!b) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_unary(%arg0: !b8i, %arg1: !p4hir.bool)
  p4hir.func @test_unary(%arg_b8i : !b8i, %arg_b : !b) {
    // CHECK: p4hir.call @blackhole (%arg0)
    %m1 = p4hir.unary(minus, %arg_b8i) : !b8i
    %m2 = p4hir.unary(minus, %m1) : !b8i
    p4hir.call @blackhole(%m2) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %c1_b8i = p4hir.unary(cmpl, %arg_b8i) : !b8i
    %c2_b8i = p4hir.unary(cmpl, %c1_b8i) : !b8i
    p4hir.call @blackhole(%c2_b8i) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole_bool (%arg1)
    %n1 = p4hir.unary(not, %arg_b) : !b
    %n2 = p4hir.unary(not, %n1) : !b
    p4hir.call @blackhole_bool(%n2) : (!b) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %p1 = p4hir.unary(plus, %arg_b8i) : !b8i
    p4hir.call @blackhole(%p1) : (!b8i) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %p2 = p4hir.unary(plus, %arg_b8i) : !b8i
    %p3 = p4hir.unary(plus, %p2) : !b8i
    p4hir.call @blackhole(%p3) : (!b8i) -> ()

    p4hir.return
  }
}
