// RUN: p4mlir-opt --canonicalize %s | FileCheck %s
!anon = !p4hir.enum<a, b>
!t = !p4hir.struct<"t", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#anon_b = #p4hir.enum_field<b, !anon> : !anon
module {
  p4hir.control @ctrl()() {
    p4hir.func action @a() {
      p4hir.return
    }
    p4hir.func action @b() {
      p4hir.return
    }
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ctrl::@a () : () -> ()
        }
        p4hir.table_action @b() {
          p4hir.call @ctrl::@b () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ctrl::@a () : () -> ()
      }
    }
    p4hir.control_apply {
      // CHECK: p4hir.table_apply
      // CHECK-NOT: p4hir.switch
      %t_apply_result = p4hir.table_apply @ctrl::@t with key() : () -> !t
      %action_run = p4hir.struct_extract %t_apply_result["action_run"] : !t
      p4hir.switch (%action_run : !anon) {
        p4hir.case(equal, [#anon_b]) {
          p4hir.yield
        }
        p4hir.case(default, []) {
          p4hir.yield
        }
        p4hir.yield
      }
    }
  }
}
