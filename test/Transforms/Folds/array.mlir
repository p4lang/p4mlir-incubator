// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>

!A = !p4hir.array<2 x !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  %c = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !A

  %idx = p4hir.const #p4hir.int<1> : !i32i
  %v = p4hir.array_get %c[%idx] : !A, !i32i

  // CHECK: %[[c20_i32i:.*]] = p4hir.const #int20_i32i
  // CHECK: p4hir.call @blackhole (%[[c20_i32i]])
  p4hir.call @blackhole(%v) : (!i32i) -> ()

  %var = p4hir.variable ["v"] : <!i32i>
  %v1 = p4hir.read %var : <!i32i>
  %array = p4hir.array [%v1, %v1] : !A
  %v2 = p4hir.array_get %array[%idx] : !A, !i32i

  // CHECK: %[[val:.*]] = p4hir.read %{{.*}} : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val]])  
  p4hir.call @blackhole(%v2) : (!i32i) -> ()
}
