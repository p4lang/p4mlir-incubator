// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b3i = !p4hir.bit<3>
!b5i = !p4hir.bit<5>
!b6i = !p4hir.bit<6>
!b8i = !p4hir.bit<8>
!i4i = !p4hir.int<4>
!i7i = !p4hir.int<7>

#int6_b3i = #p4hir.int<6> : !b3i
#int5_b5i = #p4hir.int<5> : !b5i
#int-3_i4i = #p4hir.int<-3> : !i4i

// CHECK-DAG: #[[ATTR_197:.+]] = #p4hir.int<197> : !b8i
// CHECK-DAG: #[[ATTR_M18:.+]] = #p4hir.int<-18> : !i7i
// CHECK-DAG: #[[ATTR_54:.+]] = #p4hir.int<54> : !b6i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_b6i(!b6i)
  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_i7i(!i7i)

  // CHECK-LABEL: p4hir.func @test_concat_const
  p4hir.func @test_concat_const() {
    // CHECK-DAG: %[[C197:.*]] = p4hir.const #[[ATTR_197]]
    // CHECK-DAG: %[[CM18:.*]] = p4hir.const #[[ATTR_M18]]
    // CHECK-DAG: %[[C54:.*]] = p4hir.const #[[ATTR_54]]

    %c6_b3i = p4hir.const #int6_b3i
    %c5_b5i = p4hir.const #int5_b5i
    %c_3_i4i = p4hir.const #int-3_i4i

    // CHECK: p4hir.call @blackhole_b8i (%[[C197]])
    %r1 = p4hir.concat(%c6_b3i : !b3i, %c5_b5i : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r1) : (!b8i) -> ()

    // Signed lhs: concat(0b1101, 0b110) = 0b1101110 = -18 as int<7>
    // CHECK: p4hir.call @blackhole_i7i (%[[CM18]])
    %r2 = p4hir.concat(%c_3_i4i : !i4i, %c6_b3i : !b3i) : !i7i
    p4hir.call @blackhole_i7i(%r2) : (!i7i) -> ()

    // Same-width operands: concat(0b110, 0b110) = 0b110110 = 54
    // CHECK: p4hir.call @blackhole_b6i (%[[C54]])
    %r3 = p4hir.concat(%c6_b3i : !b3i, %c6_b3i : !b3i) : !b6i
    p4hir.call @blackhole_b6i(%r3) : (!b6i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_concat_no_fold
  p4hir.func @test_concat_no_fold(%arg0: !b3i, %arg1: !b5i) {
    // CHECK: p4hir.concat
    %r = p4hir.concat(%arg0 : !b3i, %arg1 : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r) : (!b8i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_concat_partial_const
  p4hir.func @test_concat_partial_const(%arg0: !b3i) {
    %c5_b5i = p4hir.const #int5_b5i
    // CHECK: p4hir.concat
    %r = p4hir.concat(%arg0 : !b3i, %c5_b5i : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r) : (!b8i) -> ()
    p4hir.return
  }
}
