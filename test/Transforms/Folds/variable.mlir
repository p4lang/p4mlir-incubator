// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T

  %var = p4hir.variable ["v"] : <!T>
  p4hir.assign %t, %var : <!T>

  %struct = p4hir.read %var : <!T>
  %t11 = p4hir.struct_extract %struct["t1"] : !T

  // This all just simplifies down to constant
  // CHECK: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
  // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
  p4hir.call @blackhole(%t11) : (!i32i) -> ()
}
