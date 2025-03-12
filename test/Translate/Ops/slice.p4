// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: foo
action foo(in bit<32> a, inout int<8> c) {
  // CHECK: %[[s10_8:.*]] = p4hir.slice %arg0[10 : 8] : !b32i -> !b3i
  // CHECK: p4hir.slice %[[s10_8]][2 : 1] : !b3i -> !b2i
  bit<2> b = a[10:8][2:1];
  // CHECK: p4hir.slice_ref %arg1[7 : 1] : <!i8i> -> !b7i
  bit<7> d = c[7:1];

  const int e = 42;
  bit<2> f = e[1:0];
  
  bit<8> n;
  bit<8> m;
  bit<8> x;

  // CHECK: %[[n:.*]] = p4hir.variable ["n"] : <!b8i>
  // CHECK: p4hir.assign_slice %{{.*}}, %[[n]][7 : 4] : !b4i -> <!b8i>
  n[7:4][3:0][3:0] = 4w0;
  m[7:4] = 10;
  m[7:4][3:1] = 3w0;
  x[5:4][1:1] = 1w0;
}
