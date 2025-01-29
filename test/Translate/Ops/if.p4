// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

action iftest() {
   // CHECK-LABEL: module
   // CHECK: %[[B:.*]] = p4hir.alloca !p4hir.bool ["b"] : !p4hir.ref<!p4hir.bool>
   // CHECK: %[[B_VAL:.*]] = p4hir.load %[[B]] : !p4hir.ref<!p4hir.bool>, !p4hir.bool
   // CHECK: p4hir.if %[[B_VAL]] {
   // CHECK: }
   // CHECK: %[[B_VAL:.*]] = p4hir.load %0 : !p4hir.ref<!p4hir.bool>, !p4hir.bool
   // CHECK: %[[NOT_B_VAL:.*]] = p4hir.unary(not, %[[B_VAL]]) : !p4hir.bool
   // CHECK: p4hir.if %[[NOT_B_VAL]] {
   // CHECK: } else {
   // CHECK: }
    bool b;
    if (b) {

    }

    if (!b) {

    } else {
    }
}

action iftest2() {
    int<16> x1 = 1;
    if (x1 == 2) {
        int<16> x2 = 3;
    } else {
        int<16> x3 = 4;
    }
    int<16> x4 = 5;
}

action iftest3() {
    int<16> x1 = 1;
    int<16> x2 = 3;
    int<16> x3 = 4;
    if (x1 == 2) {
        x1 = 2;
        x2 = 4;
    } else if (x2 == x3) {
        x3 = 5;
        x1 = 7;
    }
    if (x1 == x2) {
        int<16> x4 = x3;
        x1 = x3;
    }
    x2 = 1;
}
