// RUN: p4mlir-opt %s | FileCheck %s

!bit32 = !p4hir.bit<32>
!i42i = !p4hir.int<42>

// CHECK: module
// CHECK-LABEL: p4hir.func action @foo(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>}, %arg1: !b32i {p4hir.dir = #p4hir<dir in>}, %arg2: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir out>}, %arg3: !i42i) {
p4hir.func action @foo(%arg0 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir inout>},
                       %arg1 : !bit32 {p4hir.dir = #p4hir<dir in>},
                       %arg2 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir out>},
                       %arg3 : !i42i) {
  %0 = p4hir.variable ["tmp"] : !p4hir.ref<!bit32>
  %1 = p4hir.read %arg0 : !p4hir.ref<!bit32>

  p4hir.assign %arg1, %0 : !p4hir.ref<!bit32>
  p4hir.assign %1, %arg2 : !p4hir.ref<!bit32>

  p4hir.implicit_return
}

p4hir.func action @bar() {
   %0 = p4hir.variable ["tmp"] : !p4hir.ref<!bit32>
   %1 = p4hir.read %0 : !p4hir.ref<!bit32>
   %3 = p4hir.const #p4hir.int<7> : !i42i
   p4hir.call @foo(%0, %1, %0, %3) : (!p4hir.ref<!bit32>, !bit32, !p4hir.ref<!bit32>, !i42i) -> ()

   p4hir.implicit_return
}
