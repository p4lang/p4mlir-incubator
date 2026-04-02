// RUN: not p4mlir-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

module {
  %0 = p4hir.const #p4hir.int<1> : !p4hir.int<8>
  %1 = p4hir.unary(not, %0) : !p4hir.int<8>
  // ERR: 'p4hir.unary' op logical not requires boolean type
}

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %1 = p4hir.unary(minus, %0) : !p4hir.bool
  // ERR: 'p4hir.unary' op arithmetic unary operations require integer-like type
}

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %1 = p4hir.unary(cmpl, %0) : !p4hir.bool
  // ERR: 'p4hir.unary' op bitwise complement operations require fixed-width integer type
}

