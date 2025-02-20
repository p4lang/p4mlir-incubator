// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

struct S {
    bit<32> f;
    bool    s;
}

const S x = { 42, false };

const tuple<bit<32>, bit<32>> t = { 0, 1 };
const bit<32> f = t[0];

// CHECK: p4hir.const ["t"] #p4hir.aggregate<[#int0_b32i, #int1_b32i]> : tuple<!b32i, !b32i>
// CHECK: p4hir.const ["f"] #int0_b32i

// CHECK-LABEL: p4hir.func action @test
action test(out bit<16> r) {
  // CHECK: %[[c10_b32i:.*]] = p4hir.const #int10_b32i
  // CHECK: %[[c12_b16i:.*]] = p4hir.const #int12_b16i
  // CHECK: %[[tuple:.*]] = p4hir.tuple (%[[c10_b32i]], %[[c12_b16i]]) : tuple<!b32i, !b16i>
  tuple<bit<32>, bit<16>> x = { 10, 12 };
  // CHECK: %[[x_0:.*]] = p4hir.variable ["x", init] : <tuple<!b32i, !b16i>>
  // CHECK: p4hir.if
  // CHECK: %[[val_4:.*]] = p4hir.read %[[x_0]] : <tuple<!b32i, !b16i>>
  // CHECK: p4hir.tuple_extract %[[val_4]][1] : tuple<!b32i, !b16i>
  if (x == { 10, 12 })
     r = x[1];
  else
     r = (bit<16>)x[0];
}

typedef tuple<bit<32>, bool> pair;
action test2() {
    pair x = { 10, false };
    tuple<bit<32>, bool> y;
    y = x;
}
