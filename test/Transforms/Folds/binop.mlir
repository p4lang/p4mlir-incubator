// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>

#int0_i32i = #p4hir.int<0> : !i32i
#int10_i32i = #p4hir.int<10> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  // CHECK-DAG: %[[c0_i32i:.*]] = p4hir.const #int0_i32i
  // CHECK-DAG: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
  %c0_i32i = p4hir.const #int0_i32i
  %c10_i32i = p4hir.const #int10_i32i

  %x = p4hir.variable ["x", init] : <!i32i>
  p4hir.assign %c10_i32i, %x : <!i32i>
  %x_val = p4hir.read %x : <!i32i>

  // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
  %res1 = p4hir.binop(add, %x_val, %c0_i32i) : !i32i
  p4hir.call @blackhole(%res1) : (!i32i) -> ()

  // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
  %res2 = p4hir.binop(add, %c0_i32i, %x_val) : !i32i
  p4hir.call @blackhole(%res2) : (!i32i) -> ()

  p4hir.call @blackhole(%c0_i32i) : (!i32i) -> ()
}

