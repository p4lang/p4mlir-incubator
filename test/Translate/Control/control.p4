// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL:   p4hir.control @Pipe1(%arg0: !b10i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "arg1"}, %arg1: !i16i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg2"}, %arg2: !p4hir.ref<!b10i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "oarg2"})()
control Pipe1(bit<10> arg1, in int<16> arg2, out bit<10> oarg2) {
// CHECK:     p4hir.func action @foo() {
    action foo() {
        int<16> x1 = 3 + arg2;
        x1 = 4;
    }
// CHECK:     p4hir.func action @bar() {    
    action bar() {
        bit<10> x1 = 2;
        x1 = x1 - arg1;
        oarg2 = x1;
    }
// CHECK:     p4hir.control_apply {    
    apply {
        bit<10> x1 = arg1;
        int<16> x2 = 5;
        x2 = arg2;
        bar();
        if (arg2 == 3) {
            foo();
            x2 = 3;
        }
    }
}
