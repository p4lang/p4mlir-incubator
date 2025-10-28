// RUN: p4mlir-opt %s | FileCheck %s

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %0 = p4hir.const #p4hir.int<42> : !p4hir.bit<8>
  %1 = p4hir.const #p4hir.int<42> : !p4hir.bit<8>

  %2 = p4hir.cmp(eq, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
  %3 = p4hir.cmp(ne, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
  %4 = p4hir.cmp(lt, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
  %5 = p4hir.cmp(le, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
  %6 = p4hir.cmp(ge, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
  %7 = p4hir.cmp(gt, %0 : !p4hir.bit<8>, %1 : !p4hir.bit<8>)
}
