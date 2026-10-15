// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -split-input-file -verify-diagnostics

!b8i = !p4hir.bit<8>

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  // expected-error@+1 {{'p4hir.cast' op cast from bool requires bool or bit<1> result, got '!p4hir.bit<8>'}}
  %1 = p4hir.cast(%0 : !p4hir.bool) : !b8i
}

// -----

!b8i = !p4hir.bit<8>

module {
  %0 = p4hir.const #p4hir.int<1> : !b8i
  // expected-error@+1 {{'p4hir.cast' op cast to bool requires bool, bit<1> or int source, got '!p4hir.bit<8>'}}
  %1 = p4hir.cast(%0 : !b8i) : !p4hir.bool
}

// -----

!i1i = !p4hir.int<1>

module {
  %0 = p4hir.const #p4hir.int<-1> : !i1i
  // expected-error@+1 {{'p4hir.cast' op cast to bool requires bool, bit<1> or int source, got '!p4hir.int<1>'}}
  %1 = p4hir.cast(%0 : !i1i) : !p4hir.bool
}

// -----

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  // expected-error@+1 {{'p4hir.cast' op cast from bool requires bool or bit<1> result, got '!p4hir.infint'}}
  %1 = p4hir.cast(%0 : !p4hir.bool) : !p4hir.infint
}

// -----

// Aliases are checked by the type they resolve to.

!b8i = !p4hir.bit<8>
!B = !p4hir.alias<"B", !p4hir.bool>

module {
  %0 = p4hir.const #p4hir.bool<true> : !B
  // expected-error@+1 {{'p4hir.cast' op cast from bool requires bool or bit<1> result, got '!p4hir.bit<8>'}}
  %1 = p4hir.cast(%0 : !B) : !b8i
}

// -----

!b8i = !p4hir.bit<8>
!B = !p4hir.alias<"B", !p4hir.bool>

module {
  %0 = p4hir.const #p4hir.int<1> : !b8i
  // expected-error@+1 {{'p4hir.cast' op cast to bool requires bool, bit<1> or int source, got '!p4hir.bit<8>'}}
  %1 = p4hir.cast(%0 : !b8i) : !B
}
