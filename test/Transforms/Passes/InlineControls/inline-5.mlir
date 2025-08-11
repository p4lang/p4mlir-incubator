// RUN: p4mlir-opt  --p4hir-inline-controls --canonicalize %s | FileCheck %s

// Inline control with out argument.

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
// CHECK-LABEL: module
module {
  p4hir.control @Callee(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "x"})() {
    %c2_b8i = p4hir.const #int2_b8i
    p4hir.control_local @__local_Callee_x_0 = %arg0 : !p4hir.ref<!b8i>
    p4hir.func action @a1() {
      %c1_b8i = p4hir.const #int1_b8i
      %__local_Callee_x_0 = p4hir.symbol_ref @Callee::@__local_Callee_x_0 : !p4hir.ref<!b8i>
      p4hir.assign %c1_b8i, %__local_Callee_x_0 : <!b8i>
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
      p4hir.assign %c2_b8i, %arg0 : <!b8i>
    }
  }
  // CHECK-LABEL: p4hir.control @Caller
  p4hir.control @Caller()() {
    // CHECK-DAG: %[[CONST_2:.*]] = p4hir.const #int2_b8i

    p4hir.instantiate @Callee () as @c

    // CHECK-DAG: %[[CALLEE_LOCAL_X_VAR:.*]] = p4hir.variable ["c.__local_Callee_x_0_var"] : <!b8i>
    // CHECK-DAG: p4hir.control_local @[[CALLEE_LOCAL_X:.*]] = %[[CALLEE_LOCAL_X_VAR]] : !p4hir.ref<!b8i>

    // Check the inlined action a1.
    // CHECK-LABEL: p4hir.func action @c.a1
    // CHECK-DAG:     %[[CONST_1:.*]] = p4hir.const #int1_b8i
    // CHECK-DAG:     %[[LOCAL_X_VAR:.*]] = p4hir.symbol_ref @Caller::@[[CALLEE_LOCAL_X]] : !p4hir.ref<!b8i>
    // CHECK-DAG:     p4hir.assign %[[CONST_1]], %[[LOCAL_X_VAR]] : <!b8i>
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

    p4hir.control_apply {
      p4hir.scope {
        // CHECK-DAG: p4hir.assign %[[CONST_2]], %[[CALLEE_LOCAL_X_VAR]] : <!b8i>
        // CHECK-DAG: %[[FINAL_X_VAL:.*]] = p4hir.read %[[CALLEE_LOCAL_X_VAR]] : <!b8i>
        // CHECK-DAG: p4hir.assign %[[FINAL_X_VAL]], %x_out_arg : <!b8i>

        %x_out_arg = p4hir.variable ["x_out_arg"] : <!b8i>
        p4hir.apply @Caller::@c(%x_out_arg) : (!p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %x_out_arg : <!b8i>
      }
    }
  }
}
