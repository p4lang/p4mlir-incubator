// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -split-input-file -verify-diagnostics

// Booleans and validity bits are not ordered.

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool
  // expected-error@+1 {{'p4hir.cmp' op only eq and ne comparisons are defined on '!p4hir.bool'}}
  %2 = p4hir.cmp(lt, %0 : !p4hir.bool, %1 : !p4hir.bool)
}

// -----

module {
  %0 = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool
  // expected-error@+1 {{'p4hir.cmp' op only eq and ne comparisons are defined on '!p4hir.bool'}}
  %2 = p4hir.cmp(ge, %0 : !p4hir.bool, %1 : !p4hir.bool)
}

// -----

!B = !p4hir.alias<"B", !p4hir.bool>

module {
  %0 = p4hir.const #p4hir.bool<true> : !B
  %1 = p4hir.const #p4hir.bool<false> : !B
  // expected-error@+1 {{'p4hir.cmp' op only eq and ne comparisons are defined on}}
  %2 = p4hir.cmp(le, %0 : !B, %1 : !B)
}

// -----

!validity_bit = !p4hir.validity.bit

module {
  %0 = p4hir.const #p4hir<validity.bit invalid> : !validity_bit
  %1 = p4hir.const #p4hir<validity.bit valid> : !validity_bit
  // expected-error@+1 {{'p4hir.cmp' op only eq and ne comparisons are defined on '!p4hir.validity.bit'}}
  %2 = p4hir.cmp(lt, %0 : !validity_bit, %1 : !validity_bit)
}

// -----

!validity_bit = !p4hir.validity.bit

module {
  %0 = p4hir.const #p4hir<validity.bit invalid> : !validity_bit
  %1 = p4hir.const #p4hir<validity.bit valid> : !validity_bit
  // expected-error@+1 {{'p4hir.cmp' op only eq and ne comparisons are defined on '!p4hir.validity.bit'}}
  %2 = p4hir.cmp(gt, %0 : !validity_bit, %1 : !validity_bit)
}
