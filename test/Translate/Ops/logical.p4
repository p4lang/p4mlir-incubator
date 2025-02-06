// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

action logical() {
// CHECK-LABEL: module
    bool flag1; bool flag2; bool flag3;

// CHECK: %[[FLAG1:.*]] = p4hir.variable ["flag1"] : <!p4hir.bool>
// CHECK: %[[FLAG2:.*]] = p4hir.variable ["flag2"] : <!p4hir.bool>
// CHECK: %[[FLAG3:.*]] = p4hir.variable ["flag3"] : <!p4hir.bool>

    bool f1 = flag2 || flag3;
// CHECK: %[[FLAG2_VAL:.*]] = p4hir.read %[[FLAG2]] : <!p4hir.bool>
// CHECK-NEXT: %[[F1_VAL:.*]] = p4hir.ternary(%[[FLAG2_VAL]], true {
// CHECK-NEXT: %[[TRUE_VAL:.*]] = p4hir.const #true
// CHECK-NEXT: p4hir.yield %[[TRUE_VAL]] : !p4hir.bool
// CHECK-NEXT:  }, false {
// CHECK-NEXT: %[[FLAG3_VAL:.*]] = p4hir.read %[[FLAG3]] : <!p4hir.bool>
// CHECK-NEXT: p4hir.yield %[[FLAG3_VAL]] : !p4hir.bool
// CHECK-NEXT  }) : (!p4hir.bool) -> !p4hir.bool    

    bool f2 = flag2 && flag3;
    bool f3 = flag2 && flag3 || flag3;
    bool f7 = flag2 || flag3 || flag3;
    bool f8 = flag2 || flag3 && flag3;
    bool f5 = flag2 || flag3;
    f5 = f1 && f5 || flag1;
    bool f6 = flag1 || flag2;
}
