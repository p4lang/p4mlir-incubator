// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

control ctrl() {
    // CHECK-LABEL: p4hir.func action @e()
    // CHECK: p4hir.exit
    action e() {
        exit;
    }

    table t {
        actions = { e; }
        default_action = e();
    }

    // CHECK-LABEL: p4hir.control_apply
    apply {
        bit<32> a;
        bit<32> b;
        bit<32> c;

        a = 0;
        b = 1;
        c = 2;
        if (t.apply().hit) {
            b = 2;
            t.apply();
            c = 3;
        } else {
            b = 3;
            t.apply();
            c = 4;
// CHECK: p4hir.exit            
            exit;
        }
        c = 5;
    }
}

control noop();
package p(noop _n);
p(ctrl()) main;
