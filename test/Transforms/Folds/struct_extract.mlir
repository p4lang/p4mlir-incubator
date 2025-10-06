// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  // CHECK: %[[var:.*]] = p4hir.variable ["t"] : <!T>
  %var = p4hir.variable ["t"] : <!T>
  %t1 = p4hir.read %var : <!T>
  %t11 = p4hir.struct_extract %t1["t1"] : !T

  // % below is to prohibit partial match to struct_extract_ref
  // CHECK-NOT: p4hir.struct_extract %
  // CHECK: %[[t1_field_ref:.*]] = p4hir.struct_extract_ref %[[var]]["t1"] : <!T>
  // CHECK: %[[val:.*]] = p4hir.read %[[t1_field_ref]] : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val]])  
  p4hir.call @blackhole(%t11) : (!i32i) -> ()

  // Multiple uses of values being read, cannot fold it here
  // CHECK: %[[var:.*]] = p4hir.variable ["t"] : <!T>
  // CHECK-NOT: p4hir.struct_extract_ref
  %var2 = p4hir.variable ["t"] : <!T>
  %t2 = p4hir.read %var2 : <!T>
  %t21 = p4hir.struct_extract %t2["t1"] : !T
  %t22 = p4hir.struct_extract %t2["t2"] : !T

  p4hir.call @blackhole(%t21) : (!i32i) -> ()
  p4hir.call @blackhole(%t22) : (!i32i) -> ()

  // Ensure that reads go before writes
  // CHECK: %[[var3:.*]] = p4hir.variable ["t3"] : <!T>
  %var3 = p4hir.variable ["t3"] : <!T>
  %t3 = p4hir.read %var3 : <!T>
  %tc = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T
  p4hir.assign %tc, %var3 : <!T>
  %t31 = p4hir.struct_extract %t3["t1"] : !T
  // CHECK: %[[t1_field_ref:.*]] = p4hir.struct_extract_ref %[[var3]]["t1"] : <!T>
  // CHECK: %[[val:.*]] = p4hir.read %[[t1_field_ref]] : <!i32i>
  // CHECK: p4hir.assign {{.*}}, %[[var3]]
  // CHECK: p4hir.call @blackhole (%[[val]])  

  p4hir.call @blackhole(%t31) : (!i32i) -> ()
}
