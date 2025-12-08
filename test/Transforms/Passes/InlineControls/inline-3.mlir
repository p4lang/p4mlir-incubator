// RUN: p4mlir-opt  --p4hir-inline-controls --canonicalize %s | FileCheck %s

// Inline control with in argument.

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i
#int4_b8i = #p4hir.int<4> : !b8i
// CHECK-LABEL: module
module {
  p4hir.control @Callee(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "x"})() {
    %c3_b8i = p4hir.const #int3_b8i
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.control_local @__local_Callee_x_0 = %arg0 : !b8i
    %add = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
    %z = p4hir.variable ["z", init] : <!b8i>
    p4hir.assign %add, %z : <!b8i>
    p4hir.control_local @__local_Callee_z_0 = %z : !p4hir.ref<!b8i>
    p4hir.func action @a1() {
      %c2_b8i = p4hir.const #int2_b8i
      %__local_Callee_z_0 = p4hir.symbol_ref @Callee::@__local_Callee_z_0 : !p4hir.ref<!b8i>
      %__local_Callee_x_0 = p4hir.symbol_ref @Callee::@__local_Callee_x_0 : !b8i
      %mul = p4hir.binop(mul, %__local_Callee_x_0, %c2_b8i) : !b8i
      p4hir.assign %mul, %__local_Callee_z_0 : <!b8i>
      p4hir.return
    }
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @a1() {
          p4hir.call @Callee::@a1 () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @Callee::@a1 () : () -> ()
      }
    }
    p4hir.control_apply {
      %val = p4hir.read %z : <!b8i>
      %add_0 = p4hir.binop(add, %val, %c3_b8i) : !b8i
      p4hir.assign %add_0, %z : <!b8i>
    }
  }
  // CHECK-LABEL: p4hir.control @Caller
  p4hir.control @Caller()() {
    // CHECK-DAG: %[[CONST_3:.*]] = p4hir.const #int3_b8i
    // CHECK-DAG: %[[CONST_4:.*]] = p4hir.const #int4_b8i
    // CHECK-DAG: %[[CONST_5:.*]] = p4hir.const #int5_b8i

    %c3_b8i = p4hir.const #int3_b8i
    %c4_b8i = p4hir.const #int4_b8i
    p4hir.instantiate @Callee () as @c
 
    // CHECK-DAG: %[[CALLEE_LOCAL_X_VAR:.*]] = p4hir.variable ["c.__local_Callee_x_0_var"] : <!b8i>
    // CHECK-DAG: p4hir.control_local @[[CALLEE_LOCAL_X:.*]] = %[[CALLEE_LOCAL_X_VAR]] : !p4hir.ref<!b8i>
    // CHECK-DAG: %[[CALLEE_LOCAL_Z_VAR:.*]] = p4hir.variable ["c.z", init] : <!b8i>
    // CHECK-DAG: p4hir.control_local @[[CALLEE_LOCAL_Z:.*]] = %[[CALLEE_LOCAL_Z_VAR]] : !p4hir.ref<!b8i>
    // CHECK-NOT: p4hir.instantiate

    // Check the inlined action a1.
    // CHECK-LABEL: p4hir.func action @c.a1
    // CHECK-DAG:     %[[CONST_2:.*]] = p4hir.const #int2_b8i
    // CHECK-DAG:     %[[LOCAL_Z_VAR:.*]] = p4hir.symbol_ref @Caller::@[[CALLEE_LOCAL_Z]] : !p4hir.ref<!b8i>
    //                X has been promoted from value to variable.
    // CHECK-DAG:     %[[LOCAL_X_VAR:.*]] = p4hir.symbol_ref @Caller::@[[CALLEE_LOCAL_X]] : !p4hir.ref<!b8i>
    //                Newly introduced read
    // CHECK-DAG:     %[[X_VAL:.*]] = p4hir.read %[[LOCAL_X_VAR]] : <!b8i>
    // CHECK-DAG:     %[[MUL:.*]] = p4hir.binop(mul, %[[X_VAL]], %[[CONST_2]]) : !b8i
    // CHECK-DAG:     p4hir.assign %[[MUL]], %[[LOCAL_Z_VAR]] : <!b8i>
    // CHECK    :     p4hir.return
    // CHECK:       }

    // CHECK-LABEL: p4hir.table @c.t
    // CHECK:         p4hir.table_actions {
    // CHECK:           p4hir.table_action @a1() {
    // CHECK:             p4hir.call @Caller::@c.a1 () : () -> ()
    // CHECK:           }
    // CHECK:         }
    // CHECK:         p4hir.table_default_action {
    // CHECK:           p4hir.call @Caller::@c.a1 () : () -> ()
    // CHECK:         }
    // CHECK:       }

    // CHECK-LABEL: p4hir.control_apply
    p4hir.control_apply {
      // CHECK-NOT: p4hir.apply
      // CHECK-DAG: p4hir.assign %[[CONST_5]], %[[CALLEE_LOCAL_Z_VAR]] : <!b8i>
      // CHECK-DAG: p4hir.assign %[[CONST_4]], %[[CALLEE_LOCAL_X_VAR]] : <!b8i>
      // CHECK-DAG: %[[LOCAL_Z_VAL:.*]] = p4hir.read %[[CALLEE_LOCAL_Z_VAR]] : <!b8i>
      // CHECK-DAG: %[[ADD:.*]] = p4hir.binop(add, %[[LOCAL_Z_VAL]], %[[CONST_3]]) : !b8i
      // CHECK-DAG: p4hir.assign %[[ADD]], %[[CALLEE_LOCAL_Z_VAR]] : <!b8i>

      p4hir.apply @Caller::@c(%c3_b8i) : (!b8i) -> ()
      p4hir.apply @Caller::@c(%c4_b8i) : (!b8i) -> ()
    }
  }
}
