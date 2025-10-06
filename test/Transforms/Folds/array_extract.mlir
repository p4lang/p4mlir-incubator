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
  %idx2 = p4hir.const #p4hir.int<2> : !i32i

  // CHECK: %[[var:.*]] = p4hir.variable ["a"]
  %var = p4hir.variable ["a"] : <!A>
  %array = p4hir.read %var : <!A>
  // CHECK-NOT: p4hir.array_get %
  // CHECK: %[[elt_ref:.*]] = p4hir.array_element_ref %[[var]][{{.*}}]
  // CHECK: %[[val:.*]] = p4hir.read %[[elt_ref]] : <!i32i>
  %v2 = p4hir.array_get %array[%idx] : !A, !i32i
  // CHECK: p4hir.call @blackhole (%[[val]])
  p4hir.call @blackhole(%v2) : (!i32i) -> ()

  // Multiple uses of values being read, cannot fold it here
  // CHECK: %[[var:.*]] = p4hir.variable ["a"]
  // CHECK-NOT: p4hir.array_element_ref
  %var2 = p4hir.variable ["a"] : <!A>
  %a2 = p4hir.read %var2 : <!A>
  %v21 = p4hir.array_get %a2[%idx] : !A, !i32i
  %v22 = p4hir.array_get %a2[%idx2] : !A, !i32i

  p4hir.call @blackhole(%v21) : (!i32i) -> ()
  p4hir.call @blackhole(%v22) : (!i32i) -> ()

  // Ensure that reads go before writes
  // CHECK: %[[var3:.*]] = p4hir.variable ["a3"]
  %var3 = p4hir.variable ["a3"] : <!A>
  %a3 = p4hir.read %var3 : <!A>
  p4hir.assign %c, %var3 : <!A>
  %v23 = p4hir.array_get %a3[%idx2] : !A, !i32i
  // CHECK: %[[elt_ref:.*]] = p4hir.array_element_ref %[[var3]][{{.*}}]
  // CHECK: %[[val:.*]] = p4hir.read %[[elt_ref]] : <!i32i>
  // CHECK: p4hir.assign {{.*}}, %[[var3]]
  // CHECK: p4hir.call @blackhole (%[[val]])  

  p4hir.call @blackhole(%v23) : (!i32i) -> ()

}
