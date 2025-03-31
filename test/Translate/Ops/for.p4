// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: p4hir.func @loop() -> !b32i {
bit<32> loop() {
    // CHECK:     %[[SUM:.*]] = p4hir.variable ["sum", init] : <!b32i>
    bit<32> sum = 0;

    // CHECK:        p4hir.scope {
    // CHECK:          %[[CONST_0:.*]] = p4hir.const #int0_b32i
    // CHECK:          %[[CAST_0:.*]] = p4hir.cast(%[[CONST_0]] : !b32i) : !b32i
    // CHECK:          %[[INDEX:.*]] = p4hir.variable ["i", init] : <!b32i>
    // CHECK:          p4hir.assign %[[CAST_0]], %[[INDEX]] : <!b32i>

    // CHECK:          p4hir.for : cond {
    // CHECK:            %[[CONST_10:.*]] = p4hir.const #int10_infint
    // CHECK:            %[[CAST_10:.*]] = p4hir.cast(%c10 : !infint) : !b32i
    // CHECK:            %[[INDEX_VAL:.*]] = p4hir.read %[[INDEX]] : <!b32i>
    // CHECK:            %[[COND:.*]] = p4hir.cmp(lt, %[[INDEX_VAL]], %[[CAST_10]]) : !b32i, !p4hir.bool
    for (bit<32> i = 0; i < 10; i = i + 1) {

        // CHECK:      } body {
        // CHECK:        %[[CONST_1:.*]] = p4hir.const #int1_b32i
        // CHECK:        %[[CURR_SUM:.*]] = p4hir.read %[[SUM]] : <!b32i>
        // CHECK:        %[[NEW_SUM:.*]] = p4hir.binop(add, %[[CURR_SUM]], %[[CONST_1]]) : !b32i
        // CHECK:        p4hir.assign %[[NEW_SUM]], %[[SUM]] : <!b32i>
        sum = sum + 1;

    }
    // CHECK:          } updates {
    // CHECK:            %[[CONST_1:.*]] = p4hir.const #int1_b32i
    // CHECK:            %[[INDEX_VAL:.*]] = p4hir.read %[[INDEX]] : <!b32i>
    // CHECK:            %[[NEW_INDEX:.*]] = p4hir.binop(add, %[[INDEX_VAL]], %[[CONST_1]]) : !b32i
    // CHECK:            p4hir.assign %[[NEW_INDEX]], %[[INDEX]] : <!b32i>
    // CHECK:          }
    // CHECK:        }

    // CHECK:     %[[FINAL_SUM:.*]] = p4hir.read %[[SUM]] : <!b32i>
    // CHECK:     p4hir.return %[[FINAL_SUM]] : !b32i
    return sum;
}

// CHECK-LABEL: p4hir.func action @multiple_statements
// CHECK:        p4hir.scope {
// CHECK:          %[[CONST_0_I:.*]] = p4hir.const #int0_b8i
// CHECK:          %[[CAST_0_I:.*]] = p4hir.cast(%[[CONST_0_I]] : !b8i) : !b8i
// CHECK:          %[[I:.*]] = p4hir.variable ["i", init] : <!b8i>
// CHECK:          p4hir.assign %[[CAST_0_I]], %[[I]] : <!b8i>
// CHECK:          %[[CONST_0_J:.*]] = p4hir.const #int0_b8i
// CHECK:          %[[CAST_0_J:.*]] = p4hir.cast(%[[CONST_0_J]] : !b8i) : !b8i
// CHECK:          %[[J:.*]] = p4hir.variable ["j", init] : <!b8i>
// CHECK:          p4hir.assign %[[CAST_0_J]], %[[J]] : <!b8i>
// CHECK:          p4hir.for : cond {
// CHECK:            %[[CONST_10:.*]] = p4hir.const #int10_infint
// CHECK:            %[[CAST_10:.*]] = p4hir.cast(%[[CONST_10]] : !infint) : !b8i
// CHECK:            %[[I_VAL:.*]] = p4hir.read %[[I]] : <!b8i>
// CHECK:            %[[CMP_I:.*]] = p4hir.cmp(le, %[[I_VAL]], %[[CAST_10]]) : !b8i, !p4hir.bool
// CHECK:            %[[TERNARY:.*]] = p4hir.ternary(%[[CMP_I]], true {
// CHECK:              %[[TRUE:.*]] = p4hir.const #true
// CHECK:              p4hir.yield %[[TRUE]] : !p4hir.bool
// CHECK:            }, false {
// CHECK:              %[[CONST_5:.*]] = p4hir.const #int5_infint
// CHECK:              %[[CAST_5:.*]] = p4hir.cast(%[[CONST_5]] : !infint) : !b8i
// CHECK:              %[[J_VAL:.*]] = p4hir.read %[[J]] : <!b8i>
// CHECK:              %[[CMP_J:.*]] = p4hir.cmp(le, %[[J_VAL]], %[[CAST_5]]) : !b8i, !p4hir.bool
// CHECK:              p4hir.yield %[[CMP_J]] : !p4hir.bool
// CHECK:            }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:          } body {
// CHECK:          } updates {
// CHECK:            %[[I_VAL_UPD:.*]] = p4hir.read %[[I]] : <!b8i>
// CHECK:            %[[J_VAL_UPD:.*]] = p4hir.read %[[J]] : <!b8i>
// CHECK:            %[[I_ADD_J:.*]] = p4hir.binop(add, %[[I_VAL_UPD]], %[[J_VAL_UPD]]) : !b8i
// CHECK:            p4hir.assign %[[I_ADD_J]], %[[I]] : <!b8i>
// CHECK:            %[[CONST_1_J:.*]] = p4hir.const #int1_b8i
// CHECK:            %[[J_VAL_UPD2:.*]] = p4hir.read %[[J]] : <!b8i>
// CHECK:            %[[J_ADD_1:.*]] = p4hir.binop(add, %[[J_VAL_UPD2]], %[[CONST_1_J]]) : !b8i
// CHECK:            p4hir.assign %[[J_ADD_1]], %[[J]] : <!b8i>
// CHECK:          }
action multiple_statements() {
    for (bit<8> i = 0, bit<8> j = 0; i <= 10 || j <= 5; i = i + j, j = j + 1) {}
}
