// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b5i = !p4hir.bit<5>
!b9i = !p4hir.bit<9>
!b11i = !p4hir.bit<11>
!i11i = !p4hir.int<11>
!int = !p4hir.infint

#int1396_int = #p4hir.int<1396> : !int
#int-652_i11i = #p4hir.int<-652> : !i11i
#int-652_b11i = #p4hir.int<1396> : !b11i

// CHECK-DAG: #[[ATTR_21:.+]] = #p4hir.int<21> : !b5i
// CHECK-DAG: #[[ATTR_372:.+]] =  #p4hir.int<372> : !b9i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_b5i(!b5i)
  p4hir.func @blackhole_b9i(!b9i)

  // CHECK: p4hir.func @test
  p4hir.func @test() {
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
}
