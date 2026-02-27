// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b1i = !p4hir.bit<1>
!b3i = !p4hir.bit<3>
!b4i = !p4hir.bit<4>
!b5i = !p4hir.bit<5>
!b6i = !p4hir.bit<6>
!b7i = !p4hir.bit<7>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!b11i = !p4hir.bit<11>
!i11i = !p4hir.int<11>
!int = !p4hir.infint

#int1396_int = #p4hir.int<1396> : !int
#int-652_i11i = #p4hir.int<-652> : !i11i
#int-652_b11i = #p4hir.int<1396> : !b11i

// CHECK-DAG: #[[ATTR_21:.+]] = #p4hir.int<21> : !b5i
// CHECK-DAG: #[[ATTR_372:.+]] =  #p4hir.int<372> : !b9i
// CHECK-DAG: #[[ATTR_14:.+]] = #p4hir.int<14> : !b4i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_b1i(!b1i)
  p4hir.func @blackhole_b3i(!b3i)
  p4hir.func @blackhole_b4i(!b4i)
  p4hir.func @blackhole_b5i(!b5i)
  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_b9i(!b9i)
  p4hir.func @blackhole_b11i(!b11i)

  // CHECK-LABEL: p4hir.func @test_const_fold
  p4hir.func @test_const_fold() {
    // CHECK-DAG: %[[CONST_21:.*]] = p4hir.const #[[ATTR_21]]
    // CHECK-DAG: %[[CONST_372:.*]] = p4hir.const #[[ATTR_372]]

    %c_infint = p4hir.const #int1396_int
    %c_int = p4hir.const #int-652_i11i
    %c_bit = p4hir.const #int-652_b11i

    // CHECK: p4hir.call @blackhole_b5i (%[[CONST_21]]) : (!b5i) -> ()
    // CHECK: p4hir.call @blackhole_b9i (%[[CONST_372]]) : (!b9i) -> ()
    %p1_infint = p4hir.slice %c_infint[10 : 6] : !int -> !b5i
    p4hir.call @blackhole_b5i(%p1_infint) : (!b5i) -> ()
    %p2_infint = p4hir.slice %c_infint[8 : 0] : !int -> !b9i
    p4hir.call @blackhole_b9i(%p2_infint) : (!b9i) -> ()

    // CHECK: p4hir.call @blackhole_b5i (%[[CONST_21]]) : (!b5i) -> ()
    // CHECK: p4hir.call @blackhole_b9i (%[[CONST_372]]) : (!b9i) -> ()
    %p1_int = p4hir.slice %c_int[10 : 6] : !i11i -> !b5i
    p4hir.call @blackhole_b5i(%p1_int) : (!b5i) -> ()
    %p2_int = p4hir.slice %c_int[8 : 0] : !i11i -> !b9i
    p4hir.call @blackhole_b9i(%p2_int) : (!b9i) -> ()

    // CHECK: p4hir.call @blackhole_b5i (%[[CONST_21]]) : (!b5i) -> ()
    // CHECK: p4hir.call @blackhole_b9i (%[[CONST_372]]) : (!b9i) -> ()
    %p1_bit = p4hir.slice %c_bit[10 : 6] : !b11i -> !b5i
    p4hir.call @blackhole_b5i(%p1_bit) : (!b5i) -> ()
    %p2_bit = p4hir.slice %c_bit[8 : 0] : !b11i -> !b9i
    p4hir.call @blackhole_b9i(%p2_bit) : (!b9i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_identity
  p4hir.func @test_slice_identity(%arg0: !b8i) {
    // CHECK-NOT: p4hir.slice
    // CHECK: p4hir.call @blackhole_b8i (%arg0)
    %r = p4hir.slice %arg0[7 : 0] : !b8i -> !b8i
    p4hir.call @blackhole_b8i(%r) : (!b8i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_identity_signed_no_fold
  p4hir.func @test_slice_identity_signed_no_fold(%arg0: !i11i) {
    // CHECK: p4hir.slice
    %r = p4hir.slice %arg0[10 : 0] : !i11i -> !b11i
    p4hir.call @blackhole_b11i(%r) : (!b11i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_compose
  p4hir.func @test_slice_compose(%arg0: !b11i) {
    // CHECK: %[[R:.*]] = p4hir.slice %arg0[7 : 4] : !b11i -> !b4i
    // CHECK: p4hir.call @blackhole_b4i (%[[R]])
    %s1 = p4hir.slice %arg0[10 : 3] : !b11i -> !b8i
    %s2 = p4hir.slice %s1[4 : 1] : !b8i -> !b4i
    p4hir.call @blackhole_b4i(%s2) : (!b4i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_compose_single_bit
  p4hir.func @test_slice_compose_single_bit(%arg0: !b11i) {
    // CHECK: %[[R:.*]] = p4hir.slice %arg0[2 : 2] : !b11i -> !b1i
    // CHECK: p4hir.call @blackhole_b1i (%[[R]])
    %s1 = p4hir.slice %arg0[8 : 2] : !b11i -> !b7i
    %s2 = p4hir.slice %s1[0 : 0] : !b7i -> !b1i
    p4hir.call @blackhole_b1i(%s2) : (!b1i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_compose_high_bits
  p4hir.func @test_slice_compose_high_bits(%arg0: !b11i) {
    // CHECK: %[[R:.*]] = p4hir.slice %arg0[10 : 8] : !b11i -> !b3i
    // CHECK: p4hir.call @blackhole_b3i (%[[R]])
    %s1 = p4hir.slice %arg0[10 : 5] : !b11i -> !b6i
    %s2 = p4hir.slice %s1[5 : 3] : !b6i -> !b3i
    p4hir.call @blackhole_b3i(%s2) : (!b3i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_compose_const
  p4hir.func @test_slice_compose_const() {
    // CHECK-DAG: %[[C14:.*]] = p4hir.const #[[ATTR_14]]
    // CHECK: p4hir.call @blackhole_b4i (%[[C14]])
    %c = p4hir.const #int-652_b11i
    %s1 = p4hir.slice %c[8 : 2] : !b11i -> !b7i
    %s2 = p4hir.slice %s1[4 : 1] : !b7i -> !b4i
    p4hir.call @blackhole_b4i(%s2) : (!b4i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_slice_no_compose
  p4hir.func @test_slice_no_compose(%arg0: !b11i) {
    // CHECK: p4hir.slice
    %s = p4hir.slice %arg0[7 : 3] : !b11i -> !b5i
    p4hir.call @blackhole_b5i(%s) : (!b5i) -> ()
    p4hir.return
  }
}
