// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: foo
action foo(in int<16> arg2, bit<10> arg1) {
    int<16> x1 = 3;
    x1 = arg2;
    bit<10> x2 = arg1;
}

// CHECK-LABEL: bar
action bar() {
    int<16> x1 = 2;
    return;
}

// CHECK-LABEL: baz
action baz(inout int<16> x) {
    x = x + 1;
    return;
}

// CHECK-LABEL: quuz
action quuz(out int<16> a) {
  a = 42;
}

// CHECK-LABEL: bazz
action bazz(in int<16> arg1) {
    // CHECK: p4hir.call @foo(%arg0, %{{.*}}) : (!i16i, !b10i) -> ()
    foo(arg1, 7);
    bit<10> x1 = 5;
    // CHECK: p4hir.call @foo(%arg0, %{{.*}}) : (!i16i, !b10i) -> ()    
    foo(arg1, x1);
    // CHECK: p4hir.call @foo(%{{.*}}, %{{.*}}) : (!i16i, !b10i) -> ()
    foo(4, 2);
    // CHECK: p4hir.call @bar() : () -> ()
    bar();
    // CHECK: p4hir.scope
    // CHECK: %[[VAR_A:.*]] = p4hir.variable ["a_out_arg"] : <!i16i>
    // CHECK: p4hir.call @quuz(%[[VAR_A]]) : (!p4hir.ref<!i16i>) -> ()
    // CHECK: p4hir.read %[[VAR_A]] : <!i16i>
    int<16> val;
    quuz(val);
    // CHECK: p4hir.scope
    // CHECK: %[[VAR_X:.*]] = p4hir.variable ["x_inout_arg", init] : <!i16i>
    // CHECK: %[[VAL_X:.*]] = p4hir.read %[[VAL:.*]] : <!i16i>
    // CHECK: p4hir.assign %[[VAL_X]], %[[VAR_X]] : <!i16i>
    // CHECK: p4hir.call @baz(%[[VAR_X]])
    // CHECK: %[[OUT_X:.*]] = p4hir.read %[[VAR_X]] : <!i16i>
    // CHECK: p4hir.assign %[[OUT_X]], %[[VAL]] : <!i16i>
    baz(val);
    return;
}
