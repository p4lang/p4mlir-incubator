// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)
  p4hir.func @blackhole_t(!T)

  // CHECK: %[[var:.*]] = p4hir.variable ["t"] : <!T>
  %var = p4hir.variable ["t"] : <!T>
  %t1 = p4hir.read %var : <!T>
  %t11 = p4hir.struct_extract %t1["t1"] : !T

  // % below is to prohibit partial match to struct_field_ref
  // CHECK-NOT: p4hir.struct_extract %
  // CHECK: %[[t1_field_ref:.*]] = p4hir.struct_field_ref %[[var]]["t1"] : <!T>
  // CHECK: %[[val:.*]] = p4hir.read %[[t1_field_ref]] : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val]])  
  p4hir.call @blackhole(%t11) : (!i32i) -> ()

  // Multiple uses of value being read, all are struct_extract.
  // CHECK: %[[var:.*]] = p4hir.variable ["t"] : <!T>
  // CHECK: %[[t2_ref:.*]] = p4hir.struct_field_ref %[[var]]["t2"] : <!T>
  // CHECK: %[[val_2:.*]] = p4hir.read %[[t2_ref]] : <!i32i>
  // CHECK: %[[t1_ref:.*]] = p4hir.struct_field_ref %[[var]]["t1"] : <!T>
  // CHECK: %[[val_1:.*]] = p4hir.read %[[t1_ref]] : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val_1]]) : (!i32i) -> ()
  // CHECK: p4hir.call @blackhole (%[[val_2]]) : (!i32i) -> ()
  // CHECK: p4hir.call @blackhole (%[[val_2]]) : (!i32i) -> ()
  %var2 = p4hir.variable ["t"] : <!T>
  %t2 = p4hir.read %var2 : <!T>
  %t21 = p4hir.struct_extract %t2["t1"] : !T
  %t22 = p4hir.struct_extract %t2["t2"] : !T
  %t22_copy = p4hir.struct_extract %t2["t2"] : !T
  p4hir.call @blackhole(%t21) : (!i32i) -> ()
  p4hir.call @blackhole(%t22) : (!i32i) -> ()
  p4hir.call @blackhole(%t22_copy) : (!i32i) -> ()

  // Multiple uses of value being read, not all are struct_extract.
  // CHECK: %{{.*}} = p4hir.variable ["t2"] : <!T>
  // CHECK-NOT: p4hir.struct_field_ref
  %var2_2 = p4hir.variable ["t2"] : <!T>
  %t2_2 = p4hir.read %var2_2 : <!T>
  %field = p4hir.struct_extract %t2_2["t1"] : !T
  p4hir.call @blackhole(%field) : (!i32i) -> ()
  p4hir.call @blackhole_t(%t2_2) : (!T) -> ()

  // Ensure that reads go before writes
  // CHECK: %[[var3:.*]] = p4hir.variable ["t3"] : <!T>
  %var3 = p4hir.variable ["t3"] : <!T>
  %t3 = p4hir.read %var3 : <!T>
  %tc = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T
  p4hir.assign %tc, %var3 : <!T>
  %t31 = p4hir.struct_extract %t3["t1"] : !T
  // CHECK: %[[t1_field_ref:.*]] = p4hir.struct_field_ref %[[var3]]["t1"] : <!T>
  // CHECK: %[[val:.*]] = p4hir.read %[[t1_field_ref]] : <!i32i>
  // CHECK: p4hir.assign {{.*}}, %[[var3]]
  // CHECK: p4hir.call @blackhole (%[[val]])  

  p4hir.call @blackhole(%t31) : (!i32i) -> ()
}
