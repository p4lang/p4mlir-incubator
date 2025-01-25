// RUN: p4mlir-opt %s | FileCheck %s

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %0 = p4hir.const #p4hir.int<42> : !p4hir.bit<8>
  %1 = p4hir.const #p4hir.int<42> : !p4hir.bit<8>

  %2 = p4hir.binop(mul, %0, %1) : !p4hir.bit<8>
  %3 = p4hir.binop(div, %0, %2) : !p4hir.bit<8>
  %4 = p4hir.binop(mod, %0, %3) : !p4hir.bit<8>
  %5 = p4hir.binop(add, %0, %4) : !p4hir.bit<8>
  %6 = p4hir.binop(sub, %0, %5) : !p4hir.bit<8>
  %7 = p4hir.binop(sadd, %0, %6) : !p4hir.bit<8>
  %8 = p4hir.binop(ssub, %0, %7) : !p4hir.bit<8>
  %9 = p4hir.binop(and, %0, %8) : !p4hir.bit<8>
  %10 = p4hir.binop(or, %0, %9) : !p4hir.bit<8>
  %11 = p4hir.binop(xor, %0, %10) : !p4hir.bit<8>  
}
