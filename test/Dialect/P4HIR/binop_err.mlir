// RUN: p4mlir-opt %s -split-input-file -verify-diagnostics

// CHECK: module
module {
  %lhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  %rhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  // expected-error @below {{'p4hir.binop' op saturating arithmetic ('sadd') is not valid for '!p4hir.infint'}}
  %sadd_int = p4hir.binop(sadd, %lhs_int, %rhs_int) : !p4hir.infint
}

// -----

// CHECK: module
module {
  %lhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  %rhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  // expected-error @below {{'p4hir.binop' op saturating arithmetic ('ssub') is not valid for '!p4hir.infint'}}
  %ssub_int = p4hir.binop(ssub, %lhs_int, %rhs_int) : !p4hir.infint
}

// -----

// CHECK: module
module {
  %lhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  %rhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  // expected-error @below {{'p4hir.binop' op bitwise operations ('and') is not valid for '!p4hir.infint'}}
  %and_int = p4hir.binop(and, %lhs_int, %rhs_int) : !p4hir.infint
}

// -----

// CHECK: module
module {
  %lhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  %rhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  // expected-error @below {{'p4hir.binop' op bitwise operations ('or') is not valid for '!p4hir.infint'}}
  %or_int = p4hir.binop(or, %lhs_int, %rhs_int) : !p4hir.infint
}

// -----

// CHECK: module
module {
  %lhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  %rhs_int = p4hir.const #p4hir.int<42> : !p4hir.infint
  // expected-error @below {{'p4hir.binop' op bitwise operations ('xor') is not valid for '!p4hir.infint'}}
  %xor_int = p4hir.binop(xor, %lhs_int, %rhs_int) : !p4hir.infint
}
