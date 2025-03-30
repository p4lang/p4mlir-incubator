// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: p4hir.func @loop() -> !b32i {
bit<32> loop() {
    // CHECK: %[[SUM:.*]] = p4hir.variable ["sum", init] : <!b32i>
    bit<32> sum = 0;

    // CHECK: p4hir.scope {
    // CHECK: %[[CONST_0:.*]] = p4hir.const #int0_b32i
    // CHECK: %[[CAST_0:.*]] = p4hir.cast(%[[CONST_0]] : !b32i) : !b32i
    // CHECK: %[[INDEX:.*]] = p4hir.variable ["i", init] : <!b32i>
    // CHECK: p4hir.assign %[[CAST_0]], %[[INDEX]] : <!b32i>

    // CHECK: p4hir.for : cond {
    // CHECK: %[[CONST_10:.*]] = p4hir.const #int10_infint
    // CHECK: %[[CAST_10:.*]] = p4hir.cast(%c10 : !infint) : !b32i
    // CHECK: %[[INDEX_VAL:.*]] = p4hir.read %[[INDEX]] : <!b32i>
    // CHECK: %[[COND:.*]] = p4hir.cmp(lt, %[[INDEX_VAL]], %[[CAST_10]]) : !b32i, !p4hir.bool
    for (bit<32> i = 0; i < 10; i = i + 1) {

        // CHECK: } body {
        // CHECK: %[[CONST_1:.*]] = p4hir.const #int1_b32i
        // CHECK: %[[CURR_SUM:.*]] = p4hir.read %[[SUM]] : <!b32i>
        // CHECK: %[[NEW_SUM:.*]] = p4hir.binop(add, %[[CURR_SUM]], %[[CONST_1]]) : !b32i
        // CHECK: p4hir.assign %[[NEW_SUM]], %[[SUM]] : <!b32i>
        sum = sum + 1;

    }
    // CHECK: } updates {
    // CHECK: %[[CONST_1:.*]] = p4hir.const #int1_b32i
    // CHECK: %[[INDEX_VAL:.*]] = p4hir.read %[[INDEX]] : <!b32i>
    // CHECK: %[[NEW_INDEX:.*]] = p4hir.binop(add, %[[INDEX_VAL]], %[[CONST_1]]) : !b32i
    // CHECK: p4hir.assign %[[NEW_INDEX]], %[[INDEX]] : <!b32i>


    //CHECK: %[[FINAL_SUM:.*]] = p4hir.read %[[SUM]] : <!b32i>
    //CHECK: p4hir.return %[[FINAL_SUM]] : !b32i

    return sum;
}
