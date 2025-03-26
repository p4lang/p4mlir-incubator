// RUN: p4mlir-opt %s | FileCheck %s

!bit32 = !p4hir.bit<32>
!i42i = !p4hir.int<42>

// CHECK: module
// CHECK-LABEL: p4hir.func action @foo(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>}, %arg1: !b32i {p4hir.dir = #p4hir<dir in>}, %arg2: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir out>}, %arg3: !i42i) {
p4hir.func action @foo(%arg0 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir inout>},
                       %arg1 : !bit32 {p4hir.dir = #p4hir<dir in>},
                       %arg2 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir out>},
                       %arg3 : !i42i) {
  %0 = p4hir.variable ["tmp"] : <!bit32>
  %1 = p4hir.read %arg0 : <!bit32>

  p4hir.assign %arg1, %0 : <!bit32>
  p4hir.assign %1, %arg2 : <!bit32>

  p4hir.return
}
