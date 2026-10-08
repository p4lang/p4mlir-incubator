// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -split-input-file -verify-diagnostics

// Lifetime markers apply to variables only.

!b8i = !p4hir.bit<8>

module {
  p4hir.func @param(%arg0 : !p4hir.ref<!b8i>) {
    // expected-error@+1 {{'p4hir.lifetime_start' op expects a variable}}
    p4hir.lifetime_start %arg0 : <!b8i>
    p4hir.return
  }
}

// -----

!b8i = !p4hir.bit<8>
!S = !p4hir.struct<"S", f: !b8i>

module {
  p4hir.func @field() {
    %s = p4hir.variable ["s"] : <!S>
    %f = p4hir.struct_field_ref %s["f"] : <!S>
    // expected-error@+1 {{'p4hir.lifetime_end' op expects a variable}}
    p4hir.lifetime_end %f : <!b8i>
    p4hir.return
  }
}
