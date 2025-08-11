// RUN: p4mlir-opt  --p4hir-inline-controls --canonicalize %s | FileCheck %s

// More complicated example.

!anon = !p4hir.enum<a1, a2>
!b32i = !p4hir.bit<32>
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>
!t = !p4hir.struct<"t", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#anon_a1 = #p4hir.enum_field<a1, !anon> : !anon
#anon_a2 = #p4hir.enum_field<a2, !anon> : !anon
#int0_b32i = #p4hir.int<0> : !b32i
#int10_b32i = #p4hir.int<10> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int50_b32i = #p4hir.int<50> : !b32i
#int54_b32i = #p4hir.int<54> : !b32i
// CHECK-LABEL: module
module {
  p4hir.extern @Y {
    p4hir.func @Y(!b32i {p4hir.dir = #undir, p4hir.param_name = "b"})
    p4hir.func @get() -> !b32i
  }
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"}, %arg1: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "y"})() {
    %c10_b32i = p4hir.const #int10_b32i
    %true = p4hir.const #true
    %c1_b32i = p4hir.const #int1_b32i
    %false = p4hir.const #false
    %c2_b32i = p4hir.const #int2_b32i
    p4hir.control_local @__local_c_x_0 = %arg0 : !p4hir.ref<!b32i>
    p4hir.control_local @__local_c_y_0 = %arg1 : !b32i
    p4hir.instantiate @Y (%c10_b32i : !b32i) as @ext
    %val = p4hir.read %arg0 : <!b32i>
    %add = p4hir.binop(add, %val, %arg1) : !b32i
    %mul = p4hir.binop(mul, %add, %c2_b32i) : !b32i
    %z = p4hir.variable ["z", init] : <!b32i>
    p4hir.assign %mul, %z : <!b32i>
    p4hir.control_local @__local_c_z_0 = %z : !p4hir.ref<!b32i>
    p4hir.func action @a1() {
      p4hir.return
    }
    p4hir.func action @a2() {
      %__local_c_z_0 = p4hir.symbol_ref @c::@__local_c_z_0 : !p4hir.ref<!b32i>
      %__local_c_z_0_0 = p4hir.symbol_ref @c::@__local_c_z_0 : !p4hir.ref<!b32i>
      %val_1 = p4hir.read %__local_c_z_0_0 : <!b32i>
      %__local_c_y_0 = p4hir.symbol_ref @c::@__local_c_y_0 : !b32i
      %add_2 = p4hir.binop(add, %val_1, %__local_c_y_0) : !b32i
      p4hir.assign %add_2, %__local_c_z_0 : <!b32i>
      p4hir.return
    }
    p4hir.func action @a3() {
      p4hir.return
    }
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @a1() {
          p4hir.call @c::@a1 () : () -> ()
        }
        p4hir.table_action @a2() {
          p4hir.call @c::@a2 () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @c::@a1 () : () -> ()
      }
    }
    p4hir.control_apply {
      %hasReturned = p4hir.variable ["hasReturned", init] : <!p4hir.bool>
      p4hir.assign %false, %hasReturned : <!p4hir.bool>
      %val_0 = p4hir.read %z : <!b32i>
      %add_1 = p4hir.binop(add, %val_0, %c1_b32i) : !b32i
      p4hir.assign %add_1, %z : <!b32i>
      %t_apply_result = p4hir.table_apply @c::@t : !t
      %action_run = p4hir.struct_extract %t_apply_result["action_run"] : !t
      p4hir.switch (%action_run : !anon) {
        p4hir.case(anyof, [#anon_a1, #anon_a2]) {
          %0 = p4hir.call_method @c::@ext::@get() : () -> !b32i
          %val_3 = p4hir.read %arg0 : <!b32i>
          %add_4 = p4hir.binop(add, %val_3, %0) : !b32i
          p4hir.assign %add_4, %arg0 : <!b32i>
          p4hir.scope {
            p4hir.assign %true, %hasReturned : <!p4hir.bool>
          }
          p4hir.yield
        }
        p4hir.case(default, []) {
          %val_3 = p4hir.read %z : <!b32i>
          %add_4 = p4hir.binop(add, %arg1, %val_3) : !b32i
          %val_5 = p4hir.read %arg0 : <!b32i>
          %add_6 = p4hir.binop(add, %val_5, %add_4) : !b32i
          p4hir.assign %add_6, %arg0 : <!b32i>
          p4hir.scope {
            p4hir.assign %true, %hasReturned : <!p4hir.bool>
          }
          p4hir.yield
        }
        p4hir.yield
      }
      %val_2 = p4hir.read %hasReturned : <!p4hir.bool>
      %not = p4hir.unary(not, %val_2) : !p4hir.bool
      p4hir.if %not {
      }
    }
  }
  // CHECK-LABEL: p4hir.control @d
  p4hir.control @d(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"})() {
    // CHECK-DAG: %[[CONST_2:.*]] = p4hir.const #int2_b32i
    // CHECK-DAG: %[[CONST_50:.*]] = p4hir.const #int50_b32i
    // CHECK-DAG: %[[CONST_54:.*]] = p4hir.const #int54_b32i

    // CHECK-DAG: %[[INST1_X_VAR:.*]] = p4hir.variable ["cinst1.__local_c_x_0_var"] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST1_X:.*]] = %[[INST1_X_VAR]] : !p4hir.ref<!b32i>
    // CHECK-DAG: %[[INST1_Y_VAR:.*]] = p4hir.variable ["cinst1.__local_c_y_0_var"] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST1_Y:.*]] = %[[INST1_Y_VAR]] : !p4hir.ref<!b32i>
    // CHECK-DAG: p4hir.instantiate @Y (%c10_b32i : !b32i) as @cinst1.ext
    // CHECK-DAG: %[[INST1_Z_VAR:.*]] = p4hir.variable ["cinst1.z", init] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST1_Z:.*]] = %[[INST1_Z_VAR]] : !p4hir.ref<!b32i>

    // CHECK: p4hir.func action @cinst1.a2() {
    // CHECK:   %[[Z_VAR_1:.*]] = p4hir.symbol_ref @d::@[[INST1_Z]] : !p4hir.ref<!b32i>
    // CHECK:   %[[Z_VAR_2:.*]] = p4hir.symbol_ref @d::@[[INST1_Z]] : !p4hir.ref<!b32i>
    // CHECK:   %[[Z_VAL:.*]] = p4hir.read %[[Z_VAR_2]] : <!b32i>
    // CHECK:   %[[Y_VAR:.*]] = p4hir.symbol_ref @d::@[[INST1_Y]] : !p4hir.ref<!b32i>
    // CHECK:   %[[Y_VAL:.*]] = p4hir.read %[[Y_VAR]] : <!b32i>
    // CHECK:   %add = p4hir.binop(add, %[[Z_VAL]], %[[Y_VAL]]) : !b32i
    // CHECK:   p4hir.assign %add, %[[Z_VAR_1]] : <!b32i>
    // CHECK:   p4hir.return
    // CHECK: }

    // CHECK-NOT: p4hir.instantiate @c

    // CHECK-DAG: %[[INST2_X_VAR:.*]] = p4hir.variable ["cinst2.__local_c_x_0_var"] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST2_X:.*]] = %[[INST2_X_VAR]] : !p4hir.ref<!b32i>
    // CHECK-DAG: %[[INST2_Y_VAR:.*]] = p4hir.variable ["cinst2.__local_c_y_0_var"] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST2_Y:.*]] = %[[INST2_Y_VAR]] : !p4hir.ref<!b32i>
    // CHECK-DAG: p4hir.instantiate @Y (%c10_b32i : !b32i) as @cinst2.ext
    // CHECK-DAG: %[[INST2_Z_VAR:.*]] = p4hir.variable ["cinst2.z", init] : <!b32i>
    // CHECK-DAG: p4hir.control_local @[[INST2_Z:.*]] = %[[INST2_Z_VAR]] : !p4hir.ref<!b32i>

    %c50_b32i = p4hir.const #int50_b32i
    %c54_b32i = p4hir.const #int54_b32i
    %c1_b32i = p4hir.const #int1_b32i
    %c2_b32i = p4hir.const #int2_b32i
    %c0_b32i = p4hir.const #int0_b32i
    p4hir.control_local @__local_d_x_0 = %arg0 : !p4hir.ref<!b32i>
    p4hir.instantiate @c () as @cinst1
    p4hir.instantiate @c () as @cinst2

    // CHECK-LABEL: p4hir.control_apply
    p4hir.control_apply {
      // CHECK: %[[INST1_APPLY_TABLE_1:.*]] = p4hir.table_apply @d::@cinst1.t : !t
      // CHECK: %[[ACTION_RUN_1:.*]] = p4hir.struct_extract %[[INST1_APPLY_TABLE_1]]["action_run"] : !t
      // CHECK: p4hir.switch (%[[ACTION_RUN_1]] : !anon) {
      //          ...
      // CHECK:   p4hir.case(default, []) {
      // CHECK:     %[[Z_VAL_1:.*]] = p4hir.read %[[INST1_Z_VAR]] : <!b32i>
      // CHECK:     %{{.*}} = p4hir.binop(add, %[[Z_VAL_1]], %[[CONST_2]]) : !b32i
      //        ...

      // CHECK: %[[INST1_APPLY_TABLE_2:.*]] = p4hir.table_apply @d::@cinst1.t : !t
      // CHECK: %[[ACTION_RUN_2:.*]] = p4hir.struct_extract %[[INST1_APPLY_TABLE_2]]["action_run"] : !t
      // CHECK: p4hir.switch (%[[ACTION_RUN_2]] : !anon) {
      //          ...
      // CHECK:   p4hir.case(default, []) {
      // CHECK:     %[[Z_VAL_2:.*]] = p4hir.read %[[INST1_Z_VAR]] : <!b32i>
      // CHECK:     %{{.*}} = p4hir.binop(add, %[[Z_VAL_2]], %[[CONST_54]]) : !b32i
      //        ...

      // CHECK: %[[INST2_APPLY_TABLE_3:.*]] = p4hir.table_apply @d::@cinst2.t : !t
      // CHECK: %[[ACTION_RUN_3:.*]] = p4hir.struct_extract %[[INST2_APPLY_TABLE_3]]["action_run"] : !t
      // CHECK: p4hir.switch (%[[ACTION_RUN_3]] : !anon) {
      //          ...
      // CHECK:   p4hir.case(default, []) {
      // CHECK:     %[[Z_VAL_3:.*]] = p4hir.read %[[INST2_Z_VAR]] : <!b32i>
      // CHECK:     %{{.*}} = p4hir.binop(add, %[[Z_VAL_3]], %[[CONST_50]]) : !b32i
      //        ...

      %y = p4hir.variable ["y", init] : <!b32i>
      p4hir.assign %c0_b32i, %y : <!b32i>
      p4hir.scope {
        %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
        %val_0 = p4hir.read %y : <!b32i>
        p4hir.assign %val_0, %x_inout_arg : <!b32i>
        p4hir.apply @d::@cinst1(%x_inout_arg, %c2_b32i) : (!p4hir.ref<!b32i>, !b32i) -> ()
        %val_1 = p4hir.read %x_inout_arg : <!b32i>
        p4hir.assign %val_1, %y : <!b32i>
      }
      %val = p4hir.read %y : <!b32i>
      %ne = p4hir.cmp(ne, %val : !b32i, %c1_b32i : !b32i)
      p4hir.if %ne {
        %val_0 = p4hir.read %arg0 : <!b32i>
        %val_1 = p4hir.read %y : <!b32i>
        %add = p4hir.binop(add, %val_0, %val_1) : !b32i
        p4hir.assign %add, %arg0 : <!b32i>
        p4hir.scope {
          %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
          %val_2 = p4hir.read %arg0 : <!b32i>
          p4hir.assign %val_2, %x_inout_arg : <!b32i>
          p4hir.apply @d::@cinst1(%x_inout_arg, %c54_b32i) : (!p4hir.ref<!b32i>, !b32i) -> ()
          %val_3 = p4hir.read %x_inout_arg : <!b32i>
          p4hir.assign %val_3, %arg0 : <!b32i>
        }
        p4hir.scope {
          %x_inout_arg = p4hir.variable ["x_inout_arg", init] : <!b32i>
          %val_2 = p4hir.read %arg0 : <!b32i>
          p4hir.assign %val_2, %x_inout_arg : <!b32i>
          p4hir.apply @d::@cinst2(%x_inout_arg, %c50_b32i) : (!p4hir.ref<!b32i>, !b32i) -> ()
          %val_3 = p4hir.read %x_inout_arg : <!b32i>
          p4hir.assign %val_3, %arg0 : <!b32i>
        }
      }
    }
  }
}
