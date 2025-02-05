// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// Adopted from testdata/p4_16_samples/pred.p4
// CHECK-LABEL: p4hir.func action @cond_0(%arg0: !p4hir.bool {p4hir.dir = #p4hir<dir undir>})
// CHECK:    %[[TMP_1:.*]] = p4hir.variable ["tmp_1"] : <!p4hir.bool>
// CHECK:    %[[TMP_2:.*]] = p4hir.variable ["tmp_2"] : <!p4hir.bool>
// CHECK:    %[[NB:.*]] = p4hir.unary(not, %arg0) : !p4hir.bool
// CHECK:    %[[MUX:.*]] = p4hir.ternary(%[[NB]], true {
// CHECK:      %[[TMP_2_VAL:.*]] = p4hir.read %[[TMP_2]] : <!p4hir.bool>
// CHECK:      p4hir.yield %[[TMP_2_VAL]] : !p4hir.bool
// CHECK:    }, false {
// CHECK:      %[[TMP_1_VAL:.*]] = p4hir.read %[[TMP_1]] : <!p4hir.bool>
// CHECK:      p4hir.yield %[[TMP_1_VAL]] : !p4hir.bool
// CHECK:    }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:    p4hir.assign %[[MUX]], %[[TMP_1]] : <!p4hir.bool>

action cond_0(bool in_b) {
   bit<32> a;
   bool tmp_1;
   bool tmp_2;

   tmp_1 = (!in_b ? tmp_2 : tmp_1);
   tmp_2 = (!in_b && !!in_b ? a == 32w5 : tmp_2);
   tmp_1 = (!in_b && !!in_b ? (!in_b && !!in_b ? a == 32w5 : tmp_2) : (!in_b && !in_b ? false : tmp_1));
}
