// RUN: p4mlir-opt --pass-pipeline='builtin.module(p4hir.control(p4hir-remove-soft-cf))' %s | FileCheck %s

!b16i = !p4hir.bit<16>
!S = !p4hir.struct<"S", f: !b16i>
#int10_b16i = #p4hir.int<10> : !b16i
#int11_b16i = #p4hir.int<11> : !b16i
#int1_b16i = #p4hir.int<1> : !b16i
#int2_b16i = #p4hir.int<2> : !b16i
module {
  p4hir.func action @NoAction() {
    p4hir.return
  }
  p4hir.control @C(%arg0: !p4hir.ref<!S> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "s"})() {
    // CHECK-DAG: %[[CONST_1:.*]] = p4hir.const #int1_b16i
    %c10_b16i = p4hir.const #int10_b16i
    %c1_b16i = p4hir.const #int1_b16i
    p4hir.control_local @__local_C_s_0 = %arg0 : !p4hir.ref<!S>
  
    // CHECK-LABEL: p4hir.func action @A
    p4hir.func action @A() {
      // CHECK-DAG: %[[CONST_2:.*]] = p4hir.const #int2_b16i
      %c11_b16i = p4hir.const #int11_b16i
      %c2_b16i = p4hir.const #int2_b16i
      %__local_C_s_0 = p4hir.symbol_ref @C::@__local_C_s_0 : !p4hir.ref<!S>
      %f_field_ref = p4hir.struct_field_ref %__local_C_s_0["f"] : <!S>
      %val = p4hir.read %f_field_ref : <!b16i>
      %gt = p4hir.cmp(gt, %val : !b16i, %c11_b16i : !b16i)

      // CHECK: p4hir.if %{{.*}} {
      // CHECK: } else {
      // CHECK:   %[[LOCAL:.*]] = p4hir.symbol_ref @C::@__local_C_s_0 : !p4hir.ref<!S>
      // CHECK:   %[[FIELD_REF:.*]] = p4hir.struct_field_ref %[[LOCAL]]["f"] : <!S>
      // CHECK:   p4hir.assign %[[CONST_2]], %[[FIELD_REF]] : <!b16i>
      // CHECK: }

      p4hir.if %gt {
        p4hir.soft_return
      }
      %__local_C_s_0_0 = p4hir.symbol_ref @C::@__local_C_s_0 : !p4hir.ref<!S>
      %f_field_ref_1 = p4hir.struct_field_ref %__local_C_s_0_0["f"] : <!S>
      p4hir.assign %c2_b16i, %f_field_ref_1 : <!b16i>
      p4hir.return
    }

    // CHECK-LABEL: p4hir.control_apply
    p4hir.control_apply {
      %f_field_ref = p4hir.struct_field_ref %arg0["f"] : <!S>
      %val = p4hir.read %f_field_ref : <!b16i>
      %gt = p4hir.cmp(gt, %val : !b16i, %c10_b16i : !b16i)

      // CHECK: p4hir.if %{{.*}} {
      // CHECK: } else {
      // CHECK:   %[[FIELD_REF:.*]] = p4hir.struct_field_ref %arg0["f"] : <!S>
      // CHECK:   p4hir.assign %[[CONST_1]], %[[FIELD_REF]] : <!b16i>
      // CHECK: }
      p4hir.if %gt {
        p4hir.soft_return
      }
      %f_field_ref_0 = p4hir.struct_field_ref %arg0["f"] : <!S>
      p4hir.assign %c1_b16i, %f_field_ref_0 : <!b16i>
    }
  }

  // Check that we don't crash.
  p4hir.control @D()() {
    p4hir.control_apply {
      p4hir.soft_return
      p4hir.exit
    }
  }
}
