// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b10i = !p4hir.bit<10>
#int1_b10i = #p4hir.int<1> : !b10i
#int10_b10i = #p4hir.int<10> : !b10i

// CHECK-DAG: #[[set_const_of_int1_b10i:.*]] = #p4hir.set<const : [#int1_b10i]> : !p4hir.set<!b10i>
// CHECK-DAG: #[[set_const_of_int1_b10i_int10_b10i:.*]] = #p4hir.set<const : [#int1_b10i, #int10_b10i]> : !p4hir.set<!b10i>

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!p4hir.set<!b10i>)
  p4hir.func @blackhole_t(!p4hir.set<tuple<!b10i, !b10i>>)

  %c1_b10i = p4hir.const #int1_b10i
  %c10_b10i = p4hir.const #int10_b10i

  // CHECK-DAG: p4hir.const #set_const_of_int1_b10i_int10_b10i
  // CHECK-DAG: p4hir.const #set_const_of_int1_b10i

  %set = p4hir.set (%c1_b10i) : !p4hir.set<!b10i>
  p4hir.call @blackhole(%set) : (!p4hir.set<!b10i>) -> ()

  %set2 = p4hir.set (%c1_b10i, %c10_b10i) : !p4hir.set<!b10i>
  p4hir.call @blackhole(%set2) : (!p4hir.set<!b10i>) -> ()

  %var = p4hir.variable ["v"] : <!b10i>
  %val = p4hir.read %var : <!b10i>
  // CHECK: %[[var:.*]] = p4hir.variable ["v"] : <!b10i>
  // CHECK: %[[val:.*]] = p4hir.read %[[var]] : <!b10i>

  // CHECK: p4hir.set (%{{.*}}, %{{.*}}, %[[val]]) : !p4hir.set<!b10i>
  %set3 = p4hir.set (%c1_b10i, %c10_b10i, %val) : !p4hir.set<!b10i>
  p4hir.call @blackhole(%set3) : (!p4hir.set<!b10i>) -> ()
}
