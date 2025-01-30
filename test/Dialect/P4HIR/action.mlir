// RUN: p4mlir-opt %s | FileCheck %s

!bit32 = !p4hir.bit<32>

// CHECK: module
// CHECK-LABEL: p4hir.action @foo(%arg0: !p4hir.ref<!p4hir.bit<32>> {p4hir.dir = #p4hir<dir inout>}, %arg1: !p4hir.bit<32> {p4hir.dir = #p4hir<dir in>}, %arg2: !p4hir.ref<!p4hir.bit<32>> {p4hir.dir = #p4hir<dir out>}, %arg3: !p4hir.int<42>) {
p4hir.action @foo(%arg0 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir inout>},
                  %arg1 : !bit32 {p4hir.dir = #p4hir<dir in>},
                  %arg2 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir out>},
                  %arg3 : !p4hir.int<42>) {
  %0 = p4hir.alloca !bit32 ["tmp"] : !p4hir.ref<!bit32>
  %1 = p4hir.load %arg0 : !p4hir.ref<!bit32>, !bit32

  p4hir.store %arg1, %0 : !bit32, !p4hir.ref<!bit32>
  p4hir.store %1, %arg2 : !bit32, !p4hir.ref<!bit32>

  p4hir.return
}
