// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL:   p4hir.func action @foo(%arg0: !b16i {p4hir.dir = #p4hir<dir in>}, %arg1: !p4hir.ref<!i10i> {p4hir.dir = #p4hir<dir inout>}, %arg2: !p4hir.ref<!b16i> {p4hir.dir = #p4hir<dir out>}, %arg3: !b16i {p4hir.dir = #p4hir<dir undir>})
// CHECK:  p4hir.return
action foo(in bit<16> arg1, inout int<10> arg2, out bit<16> arg3, bit<16> arg4) {
    bit<16> x = arg1;
    arg3 = x;
    if (arg1 == arg4) {
        arg2 = arg2 + 1;
    }
    return;
}

// CHECK-LABEL: p4hir.func action @bar(%arg0: !b16i {p4hir.dir = #p4hir<dir undir>}) {
// CHECK:  p4hir.return
action bar(bit<16> arg1) {
}
