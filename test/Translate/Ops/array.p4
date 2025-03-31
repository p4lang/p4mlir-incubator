// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

header Mpls_h {
    bit<20> label;
    bit<8> ttl;
}

struct S {
    bit<8> label;
    Mpls_h[10] mpls;
}

// FileCheck 
// CHECK: module {