// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s | FileCheck %s

!b1i = !p4hir.bit<1>
!B = !p4hir.alias<"B", !p4hir.bool>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
  %one = p4hir.const #p4hir.int<1> : !b1i
  %int_one = p4hir.const #p4hir.int<1> : !p4hir.infint

  %bit1_from_bool = p4hir.cast(%true : !p4hir.bool) : !b1i
  %bool_from_bit1 = p4hir.cast(%one : !b1i) : !p4hir.bool
  %bool_from_int = p4hir.cast(%int_one : !p4hir.infint) : !p4hir.bool
  %alias_from_bool = p4hir.cast(%true : !p4hir.bool) : !B
  %bit1_from_alias = p4hir.cast(%alias_from_bool : !B) : !b1i
}
