// RUN: p4mlir-opt %s | FileCheck %s

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %0 = p4hir.const #p4hir.int<-128> : !p4hir.int<8>
  %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool

  %2 = p4hir.unary(minus, %0) : !p4hir.int<8>
  %3 = p4hir.unary(plus, %0)  : !p4hir.int<8>
  %4 = p4hir.unary(cmpl, %0)  : !p4hir.int<8>
  %5 = p4hir.unary(not, %1)   : !p4hir.bool
}
