// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: !SimpleHeader = !p4hir.header<"SimpleHeader", len: !b8i, data: !p4hir.varbit<32>, __valid: !validity_bit>
header SimpleHeader {
  bit<8> len;
  varbit<32> data;
}

// CHECK-LABEL: p4hir.func action @test
action test(inout SimpleHeader h) {
  bit<8> src = 1;
  h.len = src;
}

