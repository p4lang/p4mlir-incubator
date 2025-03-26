// RUN: p4mlir-opt --inline %s | FileCheck %s

!action_enum = !p4hir.enum<"action_enum", b>
!b1i = !p4hir.bit<1>
!type_T = !p4hir.type_var<"T">
#undir = #p4hir<dir undir>
!m_b1i = !p4hir.package<"m"<!b1i>>
!t = !p4hir.struct<"t", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !action_enum>
#int-1_b1i = #p4hir.int<1> : !b1i
!p = !p4hir.control<"p", (!p4hir.ref<!b1i>)>
!simple_type_T = !p4hir.control<"simple"<!type_T>, (!p4hir.ref<!type_T>)>
module {
  p4hir.control @p(%arg0: !p4hir.ref<!b1i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "bt"})() {
    // CHECK-LABEL: p4hir.func action @a
    p4hir.func action @a(%arg1: !p4hir.ref<!b1i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "y0"}) {
      %c-1_b1i = p4hir.const #int-1_b1i
      %val = p4hir.read %arg1 : <!b1i>
      %or = p4hir.binop(or, %val, %c-1_b1i) : !b1i
      p4hir.assign %or, %arg1 : <!b1i>
      p4hir.return
    }
    // CHECK-LABEL: p4hir.func action @b
    // CHECK-NOT: p4hir.call
    p4hir.func action @b() {
      p4hir.scope {
        %y0_inout_arg = p4hir.variable ["y0_inout_arg", init] : <!b1i>
        %val = p4hir.read %arg0 : <!b1i>
        p4hir.assign %val, %y0_inout_arg : <!b1i>
        p4hir.call @a (%y0_inout_arg) : (!p4hir.ref<!b1i>) -> ()
        %val_0 = p4hir.read %y0_inout_arg : <!b1i>
        p4hir.assign %val_0, %arg0 : <!b1i>
      }
      p4hir.scope {
        %y0_inout_arg = p4hir.variable ["y0_inout_arg", init] : <!b1i>
        %val = p4hir.read %arg0 : <!b1i>
        p4hir.assign %val, %y0_inout_arg : <!b1i>
        p4hir.call @a (%y0_inout_arg) : (!p4hir.ref<!b1i>) -> ()
        %val_0 = p4hir.read %y0_inout_arg : <!b1i>
        p4hir.assign %val_0, %arg0 : <!b1i>
      }
      p4hir.return
    }
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @b() {
          p4hir.call @b () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @b () : () -> ()
      }
    }
    p4hir.control_apply {
      %t_apply_result = p4hir.table_apply @t : !t
    }
  }
  p4hir.package @m<[!type_T]>("pipe" : !simple_type_T {p4hir.dir = #undir, p4hir.param_name = "pipe"})
  %p = p4hir.instantiate @p() as "p" : () -> !p
  %main = p4hir.instantiate @m(%p) as "main" : (!p) -> !m_b1i
}
