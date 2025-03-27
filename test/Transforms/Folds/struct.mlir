// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T

  %t1 = p4hir.struct_extract %t["t1"] : !T

  // CHECK: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
  // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
  p4hir.call @blackhole(%t1) : (!i32i) -> ()

  %var = p4hir.variable ["v"] : <!i32i>
  %v1 = p4hir.read %var : <!i32i>
  %struct = p4hir.struct (%v1, %v1) : !T
  %t11 = p4hir.struct_extract %struct["t1"] : !T

  // CHECK: %[[val:.*]] = p4hir.read %{{.*}} : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val]])  
  p4hir.call @blackhole(%t11) : (!i32i) -> ()
}
