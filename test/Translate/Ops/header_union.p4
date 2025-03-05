// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

header H1 {
  bit<8> f;
}

header H2 {
  bit<16> g;
}

header_union U {
  H1 h1;
  H2 h2;
}

action header_union_isValid_test() {
  U u;
  if (u.isValid())
      return;
}

// CHECK: #[[$ATTR_0:.+]] = #p4hir.bool<false> : !p4hir.bool
// CHECK: #[[$ATTR_1:.+]] = #p4hir.bool<true> : !p4hir.bool
// CHECK: #[[$ATTR_2:.+]] = #p4hir<validity.bit valid> : !validity_bit
// CHECK-LABEL:   p4hir.func action @header_union_isValid_test() {
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!p4hir.header_union<"U", h1: !H1_, h2: !H2_>>
// CHECK:           %[[VAL_1:.*]] = p4hir.const #[[$ATTR_0]]
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!p4hir.header_union<"U", h1: !H1_, h2: !H2_>>
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           %[[VAL_4:.*]] = p4hir.read %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_2]]
// CHECK:           %[[VAL_6:.*]] = p4hir.cmp(eq, %[[VAL_4]], %[[VAL_5]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_7:.*]] = p4hir.ternary(%[[VAL_1]], true {
// CHECK:             %[[VAL_8:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.yield %[[VAL_8]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             p4hir.yield %[[VAL_6]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!p4hir.header_union<"U", h1: !H1_, h2: !H2_>>
// CHECK:           %[[VAL_10:.*]] = p4hir.struct_extract_ref %[[VAL_9]]["__valid"] : <!H2_>
// CHECK:           %[[VAL_11:.*]] = p4hir.read %[[VAL_10]] : <!validity_bit>
// CHECK:           %[[VAL_12:.*]] = p4hir.const #[[$ATTR_2]]
// CHECK:           %[[VAL_13:.*]] = p4hir.cmp(eq, %[[VAL_11]], %[[VAL_12]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_14:.*]] = p4hir.ternary(%[[VAL_7]], true {
// CHECK:             %[[VAL_15:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.yield %[[VAL_15]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           p4hir.if %[[VAL_14]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return
// CHECK:         }
