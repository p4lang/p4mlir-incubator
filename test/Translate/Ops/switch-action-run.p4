// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | diff -u - %s.ref
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | p4test -

// CHECK: ![[action_enum:.*]] = !p4hir.enum<a, b>
// CHECK: #[[action_enum_b:.*]] = #p4hir.enum_field<b, ![[action_enum]]> : ![[action_enum]]
// CHECK-LABEL: p4hir.control @ctrl
control ctrl() {
    action a() {}
    action b() {}

    table t {
        actions = { a; b; }
        default_action = a;
    }

    apply {
// CHECK:  p4hir.control_apply
// CHECK:      %[[t_apply_result:.*]] = p4hir.table_apply @ctrl::@t with key() : () -> !t
// CHECK:      %[[action_run:.*]] = p4hir.struct_extract %[[t_apply_result]]["action_run"] : !t
// CHECK:      p4hir.switch (%[[action_run]] : ![[action_enum]])
// CHECK:        p4hir.case(equal, [#[[action_enum_b]]])
        switch (t.apply().action_run) {
            b: { exit; }
            default: { return; }
        }
    }
}
