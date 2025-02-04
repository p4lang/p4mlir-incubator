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
    // CHECK: p4hir.call @foo(%arg0, %{{.*}}) : (!p4hir.int<16>, !p4hir.bit<10>) -> ()
    foo(arg1, 7);
    bit<10> x1 = 5;
    // CHECK: p4hir.call @foo(%arg0, %{{.*}}) : (!p4hir.int<16>, !p4hir.bit<10>) -> ()    
    foo(arg1, x1);
    // CHECK: p4hir.call @foo(%{{.*}}, %{{.*}}) : (!p4hir.int<16>, !p4hir.bit<10>) -> ()
    foo(4, 2);
    // CHECK: p4hir.call @bar() : () -> ()
    bar();
    // CHECK: p4hir.scope
    // CHECK: %[[VAR_A:.*]] = p4hir.alloca !p4hir.int<16> ["a_out"] : !p4hir.ref<!p4hir.int<16>>
    // CHECK: p4hir.call @quuz(%[[VAR_A]]) : (!p4hir.ref<!p4hir.int<16>>) -> ()
    // CHECK: p4hir.load %[[VAR_A]] : !p4hir.ref<!p4hir.int<16>>, !p4hir.int<16>
    int<16> val;
    quuz(val);
    // CHECK: p4hir.scope
    // CHECK: %[[VAR_X:.*]] = p4hir.alloca !p4hir.int<16> ["x_inout", init] : !p4hir.ref<!p4hir.int<16>>
    // CHECK: %[[VAL_X:.*]] = p4hir.load %[[VAL:.*]] : !p4hir.ref<!p4hir.int<16>>, !p4hir.int<16>
    // CHECK: p4hir.store %[[VAL_X]], %[[VAR_X]] : !p4hir.int<16>, !p4hir.ref<!p4hir.int<16>>
    // CHECK: p4hir.call @baz(%[[VAR_X]])
    // CHECK: %[[OUT_X:.*]] = p4hir.load %[[VAR_X]] : !p4hir.ref<!p4hir.int<16>>, !p4hir.int<16>
    // CHECK: p4hir.store %[[OUT_X]], %[[VAL]] : !p4hir.int<16>, !p4hir.ref<!p4hir.int<16>>
    baz(val);
    return;
}
