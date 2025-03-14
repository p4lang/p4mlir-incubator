// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: #[[$ATTR_FALSE:.+]] = #p4hir.bool<false> : !p4hir.bool
// CHECK: #[[$ATTR_TRUE:.+]] = #p4hir.bool<true> : !p4hir.bool
// CHECK: #[[$ATTR_CONST42:.+]] = #p4hir.int<42> : !b8i
// CHECK: #[[$ATTR_INVALID:.+]] = #p4hir<validity.bit invalid> : !validity_bit
// CHECK: #[[$ATTR_VALID:.+]] = #p4hir<validity.bit valid> : !validity_bit

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

// CHECK-LABEL:   p4hir.func action @header_union_isValid
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           %[[VAL_3:.*]] = p4hir.read %[[VAL_2]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:           %[[VAL_5:.*]] = p4hir.cmp(eq, %[[VAL_3]], %[[VAL_4]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_6:.*]] = p4hir.ternary(%[[VAL_5]], true {
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:             p4hir.yield %[[VAL_7]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_8]]["__valid"] : <!H2_>
// CHECK:             %[[VAL_10:.*]] = p4hir.read %[[VAL_9]] : <!validity_bit>
// CHECK:             %[[VAL_11:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:             %[[VAL_12:.*]] = p4hir.cmp(eq, %[[VAL_10]], %[[VAL_11]]) : !validity_bit, !p4hir.bool
// CHECK:             %[[VAL_13:.*]] = p4hir.ternary(%[[VAL_12]], true {
// CHECK:               %[[VAL_14:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:               p4hir.yield %[[VAL_14]] : !p4hir.bool
// CHECK:             }, false {
// CHECK:               %[[VAL_15:.*]] = p4hir.const #[[$ATTR_FALSE]]
// CHECK:               p4hir.yield %[[VAL_15]] : !p4hir.bool
// CHECK:             }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:             p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_16:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:           %[[VAL_17:.*]] = p4hir.cmp(eq, %[[VAL_6]], %[[VAL_16]]) : !p4hir.bool, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_17]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return

action header_union_isValid() {
  U u;

  if (u.isValid())
      return;
}

// CHECK-LABEL:   p4hir.func action @header_setValid
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!validity_bit>
// CHECK:           p4hir.return

action header_setValid() {
  U u;

  u.h1.setValid();
}

// CHECK-LABEL:   p4hir.func action @header_setInvalid
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           p4hir.return

action header_setInvalid() {
  U u;

  u.h1.setInvalid();
}

// CHECK-LABEL:   p4hir.func action @assign_header1
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_3]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_4]], %[[VAL_5]] : <!validity_bit>
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_7:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_6]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_7]], %[[VAL_8]] : <!validity_bit>
// CHECK:           %[[VAL_9:.*]] = p4hir.read %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_9]], %[[VAL_2]] : <!H1_>
// CHECK:           p4hir.return

action assign_header1() {
  U u;
  H1 h;

  u.h1 = h;
}

// CHECK-LABEL:   p4hir.func action @assign_header2
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["h1"] : <!H1_>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h2"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.read %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_0]] : <!H1_>
// CHECK:           p4hir.return

action assign_header2() {
  H1 h1;
  H1 h2;

  h1 = h2;
}

// CHECK-LABEL:   p4hir.func action @assign_tuple
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           %[[VAL_8:.*]] = p4hir.const #[[$ATTR_CONST42]]
// CHECK:           %[[VAL_9:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:           %[[VAL_10:.*]] = p4hir.struct (%[[VAL_8]], %[[VAL_9]]) : !H1_
// CHECK:           p4hir.assign %[[VAL_10]], %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.return

action assign_tuple() {
  U u;

  u.h1 = { 42 }; // u and u.h1 are both valid
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           p4hir.return

action assign_invalid_header1() {
  U u;

  u.h1 = (H1){#}; // u and u.h1 are both invalid
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header2
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["h"] : <!H1_>
// CHECK:           %[[VAL_1:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_1]], %[[VAL_2]] : <!validity_bit>
// CHECK:           p4hir.return

action assign_invalid_header2() {
  H1 h;

  h = (H1){#};
}

// CHECK-LABEL:   p4hir.func action @assign_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u1"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["u2"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_1]] : <!U>
// CHECK:           p4hir.return

action assign_header_union() {
  U u1;
  U u2;

  u2 = u1;
}

// CHECK-LABEL:   p4hir.func action @init_with_invalid_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u", init] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_4]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!validity_bit>
// CHECK:           p4hir.return

action init_with_invalid_header_union() {
  U u = (U){#}; // invalid header union; same as an uninitialized header union.
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_INVALID]]
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_4]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!validity_bit>
// CHECK:           p4hir.return

action assign_invalid_header_union() {
  U u;

  u = (U){#};  // invalid header union; same as an uninitialized header union.
}

// CHECK-LABEL:   p4hir.func action @get_header
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h2"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract %[[VAL_2]]["h1"] : !U
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.return

action get_header() {
  U u;
  H1 h2;

  h2 = u.h1; // get value from header union
}

// CHECK-LABEL:   p4hir.func action @equ_header_unions
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u1"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["u2"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.read %[[VAL_1]] : <!U>
// CHECK:           %[[VAL_4:.*]] = p4hir.cmp(eq, %[[VAL_2]], %[[VAL_3]]) : !U, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_4]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return

action equ_header_unions() {
  U u1;
  U u2;

  if (u1 == u2) {
    return;
  }
}

// CHECK-LABEL:   p4hir.func action @neq_header_unions
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u1"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["u2"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.read %[[VAL_1]] : <!U>
// CHECK:           %[[VAL_4:.*]] = p4hir.cmp(ne, %[[VAL_2]], %[[VAL_3]]) : !U, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_4]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return

action neq_header_unions() {
  U u1;
  U u2;

  if (u1 != u2) {
    return;
  }
}

// CHECK-LABEL:   p4hir.func action @equ_with_invalid_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract %[[VAL_1]]["h1"] : !U
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract %[[VAL_2]]["__valid"] : !H1_
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:           %[[VAL_5:.*]] = p4hir.cmp(eq, %[[VAL_3]], %[[VAL_4]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_6:.*]] = p4hir.ternary(%[[VAL_5]], true {
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:             p4hir.yield %[[VAL_7]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             %[[VAL_8:.*]] = p4hir.struct_extract %[[VAL_1]]["h2"] : !U
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_extract %[[VAL_8]]["__valid"] : !H2_
// CHECK:             %[[VAL_10:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:             %[[VAL_11:.*]] = p4hir.cmp(eq, %[[VAL_9]], %[[VAL_10]]) : !validity_bit, !p4hir.bool
// CHECK:             %[[VAL_12:.*]] = p4hir.ternary(%[[VAL_11]], true {
// CHECK:               %[[VAL_13:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:               p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:             }, false {
// CHECK:               %[[VAL_14:.*]] = p4hir.const #[[$ATTR_FALSE]]
// CHECK:               p4hir.yield %[[VAL_14]] : !p4hir.bool
// CHECK:             }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:             p4hir.yield %[[VAL_12]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_15:.*]] = p4hir.const #[[$ATTR_FALSE]]
// CHECK:           %[[VAL_16:.*]] = p4hir.cmp(eq, %[[VAL_6]], %[[VAL_15]]) : !p4hir.bool, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_16]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return

action equ_with_invalid_header_union() {
  U u;

  if (u == (U){#}) {
    return;
  }
}

// CHECK-LABEL:   p4hir.func action @neq_with_invalid_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract %[[VAL_1]]["h1"] : !U
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract %[[VAL_2]]["__valid"] : !H1_
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:           %[[VAL_5:.*]] = p4hir.cmp(eq, %[[VAL_3]], %[[VAL_4]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_6:.*]] = p4hir.ternary(%[[VAL_5]], true {
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:             p4hir.yield %[[VAL_7]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             %[[VAL_8:.*]] = p4hir.struct_extract %[[VAL_1]]["h2"] : !U
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_extract %[[VAL_8]]["__valid"] : !H2_
// CHECK:             %[[VAL_10:.*]] = p4hir.const #[[$ATTR_VALID]]
// CHECK:             %[[VAL_11:.*]] = p4hir.cmp(eq, %[[VAL_9]], %[[VAL_10]]) : !validity_bit, !p4hir.bool
// CHECK:             %[[VAL_12:.*]] = p4hir.ternary(%[[VAL_11]], true {
// CHECK:               %[[VAL_13:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:               p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:             }, false {
// CHECK:               %[[VAL_14:.*]] = p4hir.const #[[$ATTR_FALSE]]
// CHECK:               p4hir.yield %[[VAL_14]] : !p4hir.bool
// CHECK:             }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:             p4hir.yield %[[VAL_12]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_15:.*]] = p4hir.const #[[$ATTR_TRUE]]
// CHECK:           %[[VAL_16:.*]] = p4hir.cmp(eq, %[[VAL_6]], %[[VAL_15]]) : !p4hir.bool, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_16]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return

action neq_with_invalid_header_union() {
  U u;

  if (u != (U){#}) {
    return;
  }
}
