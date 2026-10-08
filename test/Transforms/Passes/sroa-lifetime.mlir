// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(any(sroa))' %s | FileCheck %s

// Lifetime markers do not block destructuring. They access no field
// themselves, and mark the lifetime of each field that remains.

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!S = !p4hir.struct<"S", a: !b8i, b: !b16i, c: !b8i>

module {
  // CHECK-LABEL: p4hir.func @split
  // CHECK-NEXT:    %[[A:.*]] = p4hir.variable ["s.field0"] : <!b8i>
  // CHECK-NEXT:    %[[B:.*]] = p4hir.variable ["s.field1"] : <!b16i>
  // CHECK-NEXT:    p4hir.lifetime_start %[[A]] : <!b8i>
  // CHECK-NEXT:    p4hir.lifetime_start %[[B]] : <!b16i>
  // CHECK-NEXT:    p4hir.assign %{{.*}}, %[[B]] : <!b16i>
  // CHECK-NEXT:    %[[V:.*]] = p4hir.read %[[A]] : <!b8i>
  // CHECK-NEXT:    p4hir.lifetime_end %[[A]] : <!b8i>
  // CHECK-NEXT:    p4hir.lifetime_end %[[B]] : <!b16i>
  // CHECK-NEXT:    p4hir.return %[[V]] : !b8i
  p4hir.func @split(%arg0 : !b16i) -> !b8i {
    %s = p4hir.variable ["s"] : <!S>
    p4hir.lifetime_start %s : <!S>
    %a = p4hir.struct_field_ref %s["a"] : <!S>
    %b = p4hir.struct_field_ref %s["b"] : <!S>
    p4hir.assign %arg0, %b : <!b16i>
    %v = p4hir.read %a : <!b8i>
    p4hir.lifetime_end %s : <!S>
    p4hir.return %v : !b8i
  }
}
