// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// Adopted from spec-ex04.p4

// CHECK-LABEL:   p4hir.func action @int_no_init
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a"] : <!i42i>
// CHECK:           p4hir.return

action int_no_init() {
    int<42> a;
}

// CHECK-LABEL:   p4hir.func action @bit_assign_bit_string_in_hex_format
// CHECK:           %[[const_int255_b32i:.*]] = p4hir.const #int255_b32i
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a", init] : <!b32i>
// CHECK:           p4hir.assign %[[const_int255_b32i]], %[[var_a]] : <!b32i>
// CHECK:           p4hir.return

action bit_assign_bit_string_in_hex_format() {
    bit<32> a = 32w0xFF;       // a 32-bit bit-string with value 00FF
}

// CHECK-LABEL:   p4hir.func action @int_assign_signed_positive_number_in_hex_format
// CHECK:           %[[const_int255_i32i:.*]] = p4hir.const #int255_i32i
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a", init] : <!i32i>
// CHECK:           p4hir.assign %[[const_int255_i32i]], %[[var_a]] : <!i32i>
// CHECK:           p4hir.return

action int_assign_signed_positive_number_in_hex_format() {
    int<32> a = 32s0xFF;       // a 32-bit signed number with value 255
}

// CHECK-LABEL:   p4hir.func action @int_assign_signed_negative_number_in_hex_format()
// CHECK:           %[[const_int_negative_255_i32i:.*]] = p4hir.const #int-255_i32i
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a", init] : <!i32i>
// CHECK:           p4hir.assign %[[const_int_negative_255_i32i]], %[[var_a]] : <!i32i>
// CHECK:           p4hir.return

action int_assign_signed_negative_number_in_hex_format() {
    int<32> a = -32s0xFF;      // a 32-bit signed number with value -255
}

// CHECK-LABEL:   p4hir.func action @bit_assign_bit_string_in_decimal_format
// CHECK:           %[[const_int_negative_86_b8i:.*]] = p4hir.const #int-86_b8i
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a", init] : <!b8i>
// CHECK:           p4hir.assign %[[const_int_negative_86_b8i]], %[[var_a]] : <!b8i>
// CHECK:           p4hir.return

action bit_assign_bit_string_in_decimal_format() {
    bit<8> a = 8w170;          // 170 (this is -86)
}

// CHECK-LABEL:   p4hir.func action @bit_assign_bit_string_in_binary_formats
// CHECK:           %[[const1_int_negative_86_b8i:.*]] = p4hir.const #int-86_b8i
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a", init] : <!b8i>
// CHECK:           p4hir.assign %[[const1_int_negative_86_b8i]], %[[var_a]] : <!b8i>
// CHECK:           %[[const2_int_negative_86_b8i:.*]] = p4hir.const #int-86_b8i
// CHECK:           %[[var_b:.*]] = p4hir.variable ["b", init] : <!b8i>
// CHECK:           p4hir.assign %[[const2_int_negative_86_b8i]], %[[var_b]] : <!b8i>
// CHECK:           %[[const3_int_negative_86_b8i:.*]] = p4hir.const #int-86_b8i
// CHECK:           %[[var_c:.*]] = p4hir.variable ["c", init] : <!b8i>
// CHECK:           p4hir.assign %[[const3_int_negative_86_b8i]], %[[var_c]] : <!b8i>
// CHECK:           p4hir.return

action bit_assign_bit_string_in_binary_formats() {
    bit<8> a = 8w0b10101010;   // an 8-bit bit-string with value AA
    bit<8> b = 8w0b_1010_1010; // same value as above
    bit<8> c = 8w0b1010_1010;  // an 8-bit unsigned number with value 170
}

// CHECK-LABEL:   p4hir.func action @init_with_cast
// CHECK:           %[[var_a:.*]] = p4hir.variable ["a"] : <!b8i>
// CHECK:           %[[val_a:.*]] = p4hir.read %[[var_a]] : <!b8i>
// CHECK:           %[[val_cast_a:.*]] = p4hir.cast(%[[val_a]] : !b8i) : !i8i
// CHECK:           %[[var_b:.*]] = p4hir.variable ["b", init] : <!i8i>
// CHECK:           p4hir.assign %[[val_cast_a]], %[[var_b]] : <!i8i>
// CHECK:           %[[val_b:.*]] = p4hir.read %[[var_b]] : <!i8i>
// CHECK:           %[[val_cast_b:.*]] = p4hir.cast(%[[val_b]] : !i8i) : !b8i
// CHECK:           %[[var_c:.*]] = p4hir.variable ["c", init] : <!b8i>
// CHECK:           p4hir.assign %[[val_cast_b]], %[[var_c]] : <!b8i>
// CHECK:           p4hir.return

action init_with_cast() {
    bit<8> a;
    int<8> b = (int<8>)a;
    bit<8> c = (bit<8>)b;
}
