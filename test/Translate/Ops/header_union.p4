// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = #p4hir.bool<false> : !p4hir.bool
// CHECK: #[[$ATTR_1:.+]] = #p4hir.bool<true> : !p4hir.bool
// CHECK: #[[$ATTR_2:.+]] = #p4hir.int<10> : !b8i
// CHECK: #[[$ATTR_3:.+]] = #p4hir.int<42> : !b8i
// CHECK: #[[$ATTR_4:.+]] = #p4hir.int<43> : !b16i
// CHECK: #[[$ATTR_5:.+]] = #p4hir<validity.bit invalid> : !validity_bit
// CHECK: #[[$ATTR_6:.+]] = #p4hir<validity.bit valid> : !validity_bit

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

// CHECK-LABEL:   p4hir.func action @isValid_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           %[[VAL_3:.*]] = p4hir.read %[[VAL_2]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_5:.*]] = p4hir.cmp(eq, %[[VAL_3]], %[[VAL_4]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_6:.*]] = p4hir.ternary(%[[VAL_5]], true {
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.yield %[[VAL_7]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_8]]["__valid"] : <!H2_>
// CHECK:             %[[VAL_10:.*]] = p4hir.read %[[VAL_9]] : <!validity_bit>
// CHECK:             %[[VAL_11:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:             %[[VAL_12:.*]] = p4hir.cmp(eq, %[[VAL_10]], %[[VAL_11]]) : !validity_bit, !p4hir.bool
// CHECK:             %[[VAL_13:.*]] = p4hir.ternary(%[[VAL_12]], true {
// CHECK:               %[[VAL_14:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:               p4hir.yield %[[VAL_14]] : !p4hir.bool
// CHECK:             }, false {
// CHECK:               %[[VAL_15:.*]] = p4hir.const #[[$ATTR_0]]
// CHECK:               p4hir.yield %[[VAL_15]] : !p4hir.bool
// CHECK:             }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:             p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_16:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:           %[[VAL_17:.*]] = p4hir.cmp(eq, %[[VAL_6]], %[[VAL_16]]) : !p4hir.bool, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_17]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return
action isValid_test() {
  U u; // U invalid

  if (u.isValid())
      return;
}

// CHECK-LABEL:   p4hir.func action @setValid_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
action setValid_test() {
  U u;

  u.h1.setValid(); // u and u.h1 are both valid
}

// CHECK-LABEL:   p4hir.func action @setInvalid_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           %[[VAL_8:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_8]], %[[VAL_9]] : <!validity_bit>
action setInvalid_test() {
  U u;

  u.h1.setInvalid(); // U invalid
}

// CHECK-LABEL:   p4hir.func action @assign_valid_header_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h1"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_3]]
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct (%[[VAL_2]], %[[VAL_3]]) : !H1_
// CHECK:           p4hir.assign %[[VAL_4]], %[[VAL_1]] : <!H1_>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_7:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_6]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_7]], %[[VAL_8]] : <!validity_bit>
// CHECK:           %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_10:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_11:.*]] = p4hir.struct_extract_ref %[[VAL_9]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_10]], %[[VAL_11]] : <!validity_bit>
// CHECK:           %[[VAL_12:.*]] = p4hir.read %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_12]], %[[VAL_5]] : <!H1_>
// CHECK:           %[[VAL_13:.*]] = p4hir.const #[[$ATTR_4]]
// CHECK:           %[[VAL_14:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_15:.*]] = p4hir.struct (%[[VAL_13]], %[[VAL_14]]) : !H2_
// CHECK:           %[[VAL_16:.*]] = p4hir.variable ["h2", init] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_15]], %[[VAL_16]] : <!H2_>
// CHECK:           %[[VAL_17:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_18:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_19:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_20:.*]] = p4hir.struct_extract_ref %[[VAL_18]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_19]], %[[VAL_20]] : <!validity_bit>
// CHECK:           %[[VAL_21:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_22:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_23:.*]] = p4hir.struct_extract_ref %[[VAL_21]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_22]], %[[VAL_23]] : <!validity_bit>
// CHECK:           %[[VAL_24:.*]] = p4hir.read %[[VAL_16]] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_24]], %[[VAL_17]] : <!H2_>
action assign_valid_header_test() {
  U u;

  H1 h1;
  h1 = { 42 };
  u.h1 = h1; // u and u.h1 are both valid

  H2 h2 = { 43 };
  u.h2 = h2; // u and u.h2 are both valid
}

// CHECK-LABEL:   p4hir.func action @assign_tuple_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_2]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_3]], %[[VAL_4]] : <!validity_bit>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           %[[VAL_8:.*]] = p4hir.const #[[$ATTR_2]]
// CHECK:           %[[VAL_9:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_10:.*]] = p4hir.struct (%[[VAL_8]], %[[VAL_9]]) : !H1_
// CHECK:           p4hir.assign %[[VAL_10]], %[[VAL_1]] : <!H1_>
action assign_tuple_test() {
  U u;

  u.h1 = { 10 }; // u and u.h1 are both valid
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_7:.*]] = p4hir.struct_extract_ref %[[VAL_5]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_6]], %[[VAL_7]] : <!validity_bit>
// CHECK:           %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_9:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_10:.*]] = p4hir.struct_extract_ref %[[VAL_8]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_9]], %[[VAL_10]] : <!validity_bit>
// CHECK:           %[[VAL_11:.*]] = p4hir.read %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_11]], %[[VAL_4]] : <!H1_>
action assign_invalid_header_test() {
  U u;

  H1 h;
  h = (H1){#}; // Make h invalid

  u.h1 = h; // u and u.h1 are both invalid
}

// CHECK-LABEL:   p4hir.func action @init_invalid_header_union_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u", init] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_4]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!validity_bit>
action init_invalid_header_union_test() {
  U u = (U){#}; // invalid header union; same as an uninitialized header union.
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header_union_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract_ref %[[VAL_1]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!validity_bit>
// CHECK:           %[[VAL_4:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_5:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_4]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!validity_bit>
action assign_invalid_header_union_test() {
  U u;

  u = (U){#};  // invalid header union; same as an uninitialized header union.
}

// CHECK-LABEL:   p4hir.func action @get_header_value_test
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.variable ["h1"] : <!H1_>
// CHECK:           %[[VAL_2:.*]] = p4hir.const #[[$ATTR_2]]
// CHECK:           %[[VAL_3:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_4:.*]] = p4hir.struct (%[[VAL_2]], %[[VAL_3]]) : !H1_
// CHECK:           p4hir.assign %[[VAL_4]], %[[VAL_1]] : <!H1_>
// CHECK:           %[[VAL_5:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_6:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h1"] : <!U>
// CHECK:           %[[VAL_7:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_8:.*]] = p4hir.struct_extract_ref %[[VAL_6]]["__valid"] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_7]], %[[VAL_8]] : <!validity_bit>
// CHECK:           %[[VAL_9:.*]] = p4hir.struct_extract_ref %[[VAL_0]]["h2"] : <!U>
// CHECK:           %[[VAL_10:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK:           %[[VAL_11:.*]] = p4hir.struct_extract_ref %[[VAL_9]]["__valid"] : <!H2_>
// CHECK:           p4hir.assign %[[VAL_10]], %[[VAL_11]] : <!validity_bit>
// CHECK:           %[[VAL_12:.*]] = p4hir.read %[[VAL_1]] : <!H1_>
// CHECK:           p4hir.assign %[[VAL_12]], %[[VAL_5]] : <!H1_>
// CHECK:           %[[VAL_13:.*]] = p4hir.variable ["h2"] : <!H1_>
// CHECK:           %[[VAL_14:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_15:.*]] = p4hir.struct_extract %[[VAL_14]]["h1"] : !U
// CHECK:           p4hir.assign %[[VAL_15]], %[[VAL_13]] : <!H1_>
action get_header_value_test() {
  U u;

  H1 h1;
  h1 = { 10 };

  u.h1 = h1; // u and u.h1 are both valid

  H1 h2;
  h2 = u.h1; // get value from header union
}

// CHECK-LABEL:   p4hir.func action @compare_with_invalid_header_union
// CHECK:           %[[VAL_0:.*]] = p4hir.variable ["u"] : <!U>
// CHECK:           %[[VAL_1:.*]] = p4hir.read %[[VAL_0]] : <!U>
// CHECK:           %[[VAL_2:.*]] = p4hir.struct_extract %[[VAL_1]]["h1"] : !U
// CHECK:           %[[VAL_3:.*]] = p4hir.struct_extract %[[VAL_2]]["__valid"] : !H1_
// CHECK:           %[[VAL_4:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:           %[[VAL_5:.*]] = p4hir.cmp(eq, %[[VAL_3]], %[[VAL_4]]) : !validity_bit, !p4hir.bool
// CHECK:           %[[VAL_6:.*]] = p4hir.ternary(%[[VAL_5]], true {
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.yield %[[VAL_7]] : !p4hir.bool
// CHECK:           }, false {
// CHECK:             %[[VAL_8:.*]] = p4hir.struct_extract %[[VAL_1]]["h2"] : !U
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_extract %[[VAL_8]]["__valid"] : !H2_
// CHECK:             %[[VAL_10:.*]] = p4hir.const #[[$ATTR_6]]
// CHECK:             %[[VAL_11:.*]] = p4hir.cmp(eq, %[[VAL_9]], %[[VAL_10]]) : !validity_bit, !p4hir.bool
// CHECK:             %[[VAL_12:.*]] = p4hir.ternary(%[[VAL_11]], true {
// CHECK:               %[[VAL_13:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:               p4hir.yield %[[VAL_13]] : !p4hir.bool
// CHECK:             }, false {
// CHECK:               %[[VAL_14:.*]] = p4hir.const #[[$ATTR_0]]
// CHECK:               p4hir.yield %[[VAL_14]] : !p4hir.bool
// CHECK:             }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:             p4hir.yield %[[VAL_12]] : !p4hir.bool
// CHECK:           }) : (!p4hir.bool) -> !p4hir.bool
// CHECK:           %[[VAL_15:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:           %[[VAL_16:.*]] = p4hir.cmp(eq, %[[VAL_6]], %[[VAL_15]]) : !p4hir.bool, !p4hir.bool
// CHECK:           p4hir.if %[[VAL_16]] {
// CHECK:             p4hir.return
// CHECK:           }
// CHECK:           p4hir.return
action compare_with_invalid_header_union() {
  U u;

  if (u == (U){#}) { // This is equivalent to the condition !u.isValid()
    return;
  }
}
