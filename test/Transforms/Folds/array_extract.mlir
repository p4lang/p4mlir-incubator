// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!b32i = !p4hir.bit<32>

!A = !p4hir.array<2 x !i32i>

#int1_i32i = #p4hir.int<1> : !i32i
#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  // CHECK-DAG: %[[CONST_1:.*]] = p4hir.const #int1_i32i

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

  // CHECK-DAG: %[[ARR_VAR:.*]] = p4hir.variable ["arr"] : <!arr_2xi32i>
  // CHECK-DAG: %[[IDX_VAR:.*]] = p4hir.variable ["idx"] : <!i32i>
  %arr = p4hir.variable ["arr"] : <!A>
  %arr_idx = p4hir.variable ["idx"] : <!i32i>

  // Check that we can't canonicalize to ref when index is after read.
  // CHECK-DAG: %[[ARR1:.*]] = p4hir.read %[[ARR_VAR]] : <!arr_2xi32i>
  // CHECK-DAG: %[[IDX1:.*]] = p4hir.read %[[IDX_VAR]] : <!i32i>
  // CHECK-DAG: %[[RES1:.*]] = p4hir.array_get %val_5[%val_6] : !arr_2xi32i, !i32i
  // CHECK-DAG: p4hir.call @blackhole (%[[RES1]]) : (!i32i) -> ()
  %val_0 = p4hir.read %arr : <!A>
  %val_1 = p4hir.read %arr_idx : <!i32i>
  %array_elt_1 = p4hir.array_get %val_0[%val_1] : !A, !i32i
  p4hir.call @blackhole(%array_elt_1) : (!i32i) -> ()

  // Check that we can canonicalize to ref when index is before read in the same block.
  // CHECK-DAG: %[[IDX2:.*]] = p4hir.read %[[IDX_VAR]] : <!i32i>
  // CHECK-DAG: %[[ARR_REF2:.*]] = p4hir.array_element_ref %[[ARR_VAR]][%[[IDX2]]] : !p4hir.ref<!arr_2xi32i>, !i32i
  // CHECK-DAG: %[[RES2:.*]] = p4hir.read %[[ARR_REF2]] : <!i32i>
  // CHECK-DAG: p4hir.call @blackhole (%[[RES2]]) : (!i32i) -> ()
  %val_2 = p4hir.read %arr_idx : <!i32i>
  %val_3 = p4hir.read %arr : <!A>
  %array_elt_2 = p4hir.array_get %val_3[%val_2] : !A, !i32i
  p4hir.call @blackhole(%array_elt_2) : (!i32i) -> ()

  // Check with index defined in other block that dominates the read.
  %cst = p4hir.const #int1_i32i
  %dummy_cond = p4hir.cmp(eq, %val_2 : !i32i, %cst : !i32i)
  p4hir.if %dummy_cond {
    // CHECK-DAG: %[[ARRAY_REF3:.*]] = p4hir.array_element_ref %arr[%[[CONST_1]]] : !p4hir.ref<!arr_2xi32i>, !i32i
    // CHECK-DAG: %[[RES3:.*]] = p4hir.read %[[ARRAY_REF3]] : <!i32i>
    // CHECK-DAG: p4hir.call @blackhole (%[[RES3]]) : (!i32i) -> ()
    %val_4 = p4hir.read %arr : <!A>
    %array_elt_3 = p4hir.array_get %val_4[%cst] : !A, !i32i
    p4hir.call @blackhole(%array_elt_3) : (!i32i) -> ()
  }
}
