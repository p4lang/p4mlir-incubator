// RUN: p4mlir-opt  --p4hir-inline-controls --canonicalize %s | FileCheck %s

// Test with constructor arguments.

!b32i = !p4hir.bit<32>
#undir = #p4hir<dir undir>
#Callee1_val = #p4hir.ctor_param<@Callee1, "val"> : !b32i
#int0_b32i = #p4hir.int<0> : !b32i
#int10_b32i = #p4hir.int<10> : !b32i
module {
  p4hir.extern @Y {
    p4hir.func @Y(!b32i {p4hir.dir = #undir, p4hir.param_name = "b"})
    p4hir.func @get(!b32i {p4hir.dir = #undir, p4hir.param_name = "x"}) -> !b32i
  }
  p4hir.control @Callee1(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"})(val: !b32i) {
    %val = p4hir.const ["val"] #Callee1_val
    p4hir.control_local @__local_Callee1_x_0 = %arg0 : !p4hir.ref<!b32i>
    p4hir.instantiate @Y (%val : !b32i) as @ext
    p4hir.control_apply {
      %0 = p4hir.call_method @Y::@get (%val) of @Callee1::@ext : (!b32i) -> !b32i
    }
  }
  // CHECK-LABEL: p4hir.control @Callee2
  p4hir.control @Callee2(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"})() {
    // CHECK-DAG: %[[CONST_0:.*]] = p4hir.const #int0_b32i
    // CHECK-DAG: %[[CONST_10:.*]] = p4hir.const #int10_b32i
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_0]] : !b32i) as @c1.ext
    // CHECK-DAG: p4hir.instantiate @Y (%[[CONST_10]] : !b32i) as @c2.ext
    %c10_b32i = p4hir.const #int10_b32i
    %c0_b32i = p4hir.const #int0_b32i
    p4hir.control_local @__local_Callee2_x_0 = %arg0 : !p4hir.ref<!b32i>
    p4hir.instantiate @Callee1 (%c0_b32i : !b32i) as @c1
    p4hir.instantiate @Callee1 (%c10_b32i : !b32i) as @c2
    // CHECK: p4hir.control_apply
    p4hir.control_apply {
      // CHECK-DAG: %{{.*}} = p4hir.call_method @Y::@get (%[[CONST_0]]) of @Callee2::@c1.ext : (!b32i) -> !b32i
      // CHECK-DAG: %{{.*}} = p4hir.call_method @Y::@get (%[[CONST_10]]) of @Callee2::@c2.ext : (!b32i) -> !b32i
      p4hir.scope {
        %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
        %val = p4hir.read %arg0 : <!b32i>
        p4hir.assign %val, %x_inout_arg : <!b32i>
        p4hir.apply @Callee2::@c1(%x_inout_arg) : (!p4hir.ref<!b32i>) -> ()
        %val_0 = p4hir.read %x_inout_arg : <!b32i>
        p4hir.assign %val_0, %arg0 : <!b32i>
      }
      p4hir.scope {
        %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
        %val = p4hir.read %arg0 : <!b32i>
        p4hir.assign %val, %x_inout_arg : <!b32i>
        p4hir.apply @Callee2::@c2(%x_inout_arg) : (!p4hir.ref<!b32i>) -> ()
        %val_0 = p4hir.read %x_inout_arg : <!b32i>
        p4hir.assign %val_0, %arg0 : <!b32i>
      }
    }
  }
}
