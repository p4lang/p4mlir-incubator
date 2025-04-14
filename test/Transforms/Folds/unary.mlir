// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b8 = !p4hir.bit<8>
!b  = !p4hir.bool

#int5_b8 = #p4hir.int<5> : !b8
#true_b = #p4hir.bool<true> : !b
#false_b = #p4hir.bool<false> : !b

module {
  p4hir.func @blackhole(!b8)
  p4hir.func @blackhole_bool(!b)

  // CHECK-LABEL: p4hir.func @test_unary_const
  p4hir.func @test_unary_const() {
    // CHECK: p4hir.call @blackhole (%{{.*}}-5{{.*}}) : (!b8i) -> ()
    %c1 = p4hir.const #int5_b8
    %r1 = p4hir.unary(minus, %c1) : !b8
    p4hir.call @blackhole(%r1) : (!b8) -> ()

    // CHECK: p4hir.call @blackhole (%{{.*}}5{{.*}}) : (!b8i) -> ()
    %c2 = p4hir.const #int5_b8
    %r2 = p4hir.unary(plus, %c2) : !b8
    p4hir.call @blackhole(%r2) : (!b8) -> ()

    // ~5 = -6
    // CHECK: p4hir.call @blackhole (%{{.*}}-6{{.*}}) : (!b8i) -> ()
    %c3 = p4hir.const #int5_b8
    %r3 = p4hir.unary(cmpl, %c3) : !b8
    p4hir.call @blackhole(%r3) : (!b8) -> ()

    // CHECK: p4hir.call @blackhole_bool (%{{.*}}false{{.*}}) : (!p4hir.bool) -> ()
    %c4 = p4hir.const #true_b
    %r4 = p4hir.unary(not, %c4) : !b
    p4hir.call @blackhole_bool(%r4) : (!b) -> ()

    // CHECK: p4hir.call @blackhole_bool (%{{.*}}true{{.*}}) : (!p4hir.bool) -> ()
    %c5 = p4hir.const #false_b
    %r5 = p4hir.unary(not, %c5) : !b
    p4hir.call @blackhole_bool(%r5) : (!b) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_unary(%arg0: !b8i, %arg1: !p4hir.bool)
  p4hir.func @test_unary(%arg_b8 : !b8, %arg_b : !b) {
    // CHECK: p4hir.call @blackhole (%arg0)
    %m1 = p4hir.unary(minus, %arg_b8) : !b8
    %m2 = p4hir.unary(minus, %m1) : !b8
    p4hir.call @blackhole(%m2) : (!b8) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %c1 = p4hir.unary(cmpl, %arg_b8) : !b8
    %c2 = p4hir.unary(cmpl, %c1) : !b8
    p4hir.call @blackhole(%c2) : (!b8) -> ()

    // CHECK: p4hir.call @blackhole_bool (%arg1)
    %n1 = p4hir.unary(not, %arg_b) : !b
    %n2 = p4hir.unary(not, %n1) : !b
    p4hir.call @blackhole_bool(%n2) : (!b) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %p1 = p4hir.unary(plus, %arg_b8) : !b8
    p4hir.call @blackhole(%p1) : (!b8) -> ()

    // CHECK: p4hir.call @blackhole (%arg0)
    %p2 = p4hir.unary(plus, %arg_b8) : !b8
    %p3 = p4hir.unary(plus, %p2) : !b8
    p4hir.call @blackhole(%p3) : (!b8) -> ()

    p4hir.return
  }
}
