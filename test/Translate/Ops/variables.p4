// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// Adopted from spec-ex04.p4

action foo() {
  bit<32> b0 = 32w0xFF;       // a 32-bit bit-string with value 00FF
  int<32> b2 = 32s0xFF;       // a 32-bit signed number with value 255
  int<32> b3 = -32s0xFF;      // a 32-bit signed number with value -255
  bit<8> b4 = 8w0b10101010;   // an 8-bit bit-string with value AA
  bit<8> b5 = 8w0b_1010_1010; // same value as above
  bit<8> b6 = 8w170;          // same value as above
  bit<8> b7 = 8w0b1010_1010;  // an 8-bit unsigned number with value 170
  int<8> b8 = (int<8>)b7;
  int<42> b9;
  bit<8> b10 = (bit<8>)b8;
}

// CHECK-LABEL: p4hir.func action @foo()   
// CHECK:         %[[VAL_0:.*]] = p4hir.const #p4hir.int<255> : !p4hir.bit<32>
// CHECK:         %[[VAL_1:.*]] = p4hir.alloca !p4hir.bit<32> ["b0", init] : !p4hir.ref<!p4hir.bit<32>>
// CHECK:         p4hir.store %[[VAL_0]], %[[VAL_1]] : !p4hir.bit<32>, !p4hir.ref<!p4hir.bit<32>>
// CHECK:         %[[VAL_2:.*]] = p4hir.const #p4hir.int<255> : !p4hir.int<32>
// CHECK:         %[[VAL_3:.*]] = p4hir.alloca !p4hir.int<32> ["b2", init] : !p4hir.ref<!p4hir.int<32>>
// CHECK:         p4hir.store %[[VAL_2]], %[[VAL_3]] : !p4hir.int<32>, !p4hir.ref<!p4hir.int<32>>
// CHECK:         %[[VAL_4:.*]] = p4hir.const #p4hir.int<-255> : !p4hir.int<32>
// CHECK:         %[[VAL_5:.*]] = p4hir.alloca !p4hir.int<32> ["b3", init] : !p4hir.ref<!p4hir.int<32>>
// CHECK:         p4hir.store %[[VAL_4]], %[[VAL_5]] : !p4hir.int<32>, !p4hir.ref<!p4hir.int<32>>
// CHECK:         %[[VAL_6:.*]] = p4hir.const #p4hir.int<170> : !p4hir.bit<8>
// CHECK:         %[[VAL_7:.*]] = p4hir.alloca !p4hir.bit<8> ["b4", init] : !p4hir.ref<!p4hir.bit<8>>
// CHECK:         p4hir.store %[[VAL_6]], %[[VAL_7]] : !p4hir.bit<8>, !p4hir.ref<!p4hir.bit<8>>
// CHECK:         %[[VAL_8:.*]] = p4hir.const #p4hir.int<170> : !p4hir.bit<8>
// CHECK:         %[[VAL_9:.*]] = p4hir.alloca !p4hir.bit<8> ["b5", init] : !p4hir.ref<!p4hir.bit<8>>
// CHECK:         p4hir.store %[[VAL_8]], %[[VAL_9]] : !p4hir.bit<8>, !p4hir.ref<!p4hir.bit<8>>
// CHECK:         %[[VAL_10:.*]] = p4hir.const #p4hir.int<170> : !p4hir.bit<8>
// CHECK:         %[[VAL_11:.*]] = p4hir.alloca !p4hir.bit<8> ["b6", init] : !p4hir.ref<!p4hir.bit<8>>
// CHECK:         p4hir.store %[[VAL_10]], %[[VAL_11]] : !p4hir.bit<8>, !p4hir.ref<!p4hir.bit<8>>
// CHECK:         %[[VAL_12:.*]] = p4hir.const #p4hir.int<170> : !p4hir.bit<8>
// CHECK:         %[[VAL_13:.*]] = p4hir.alloca !p4hir.bit<8> ["b7", init] : !p4hir.ref<!p4hir.bit<8>>
// CHECK:         p4hir.store %[[VAL_12]], %[[VAL_13]] : !p4hir.bit<8>, !p4hir.ref<!p4hir.bit<8>>
// CHECK:         %[[VAL_14_1:.*]] = p4hir.load %[[VAL_13]] : !p4hir.ref<!p4hir.bit<8>>, !p4hir.bit<8>
// CHECK:         %[[VAL_15_1:.*]] = p4hir.cast(%[[VAL_14_1]] : !p4hir.bit<8>) : !p4hir.int<8>
// CHECK:         %[[VAL_16_1:.*]] = p4hir.alloca !p4hir.int<8> ["b8", init] : !p4hir.ref<!p4hir.int<8>>
// CHECK:         p4hir.store %[[VAL_15_1]], %[[VAL_16_1]] : !p4hir.int<8>, !p4hir.ref<!p4hir.int<8>>
// CHECK:         %[[VAL_14:.*]] = p4hir.alloca !p4hir.int<42> ["b9"] : !p4hir.ref<!p4hir.int<42>>
// CHECK:         %[[VAL_14_2:.*]] = p4hir.load %[[VAL_16_1]] : !p4hir.ref<!p4hir.int<8>>, !p4hir.int<8>
// CHECK:         %[[VAL_16:.*]] = p4hir.cast(%[[VAL_14_2]] : !p4hir.int<8>) : !p4hir.bit<8>
// CHECK:         %[[VAL_15:.*]] = p4hir.alloca !p4hir.bit<8> ["b10", init] : !p4hir.ref<!p4hir.bit<8>>
// CHECK:         p4hir.store %[[VAL_16]], %[[VAL_15]] : !p4hir.bit<8>, !p4hir.ref<!p4hir.bit<8>>
