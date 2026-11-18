// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(any(sroa))' < %s | FileCheck %s

!b4i = !p4hir.bit<4>
!b5i = !p4hir.bit<5>
!S = !p4hir.struct<"S", a: !b5i, b: !b4i>

module {
  // CHECK-LABEL: p4hir.func @read_slice_of_field
  // CHECK-NOT:     p4hir.variable ["s", init] : <!S>
  // CHECK:         %[[A:.*]] = p4hir.variable ["s.field0", init] : <!b5i>
  // CHECK:         p4hir.read_slice %[[A]][4 : 1] : <!b5i> -> !b4i
  p4hir.func @read_slice_of_field(%arg0 : !b5i, %arg1 : !b4i) -> !b4i {
    %s = p4hir.variable ["s", init] : <!S>
    %s_val = p4hir.struct (%arg0, %arg1) : !S
    p4hir.assign %s_val, %s : <!S>
    %a_ref = p4hir.struct_field_ref %s["a"] : <!S>
    %slice = p4hir.read_slice %a_ref[4 : 1] : <!b5i> -> !b4i
    p4hir.return %slice : !b4i
  }

  // CHECK-LABEL: p4hir.func @assign_slice_of_field
  // CHECK-NOT:     p4hir.variable ["s", init] : <!S>
  // CHECK:         %[[A:.*]] = p4hir.variable ["s.field0", init] : <!b5i>
  // CHECK:         p4hir.assign_slice %{{.*}}, %[[A]][4 : 1] : !b4i -> <!b5i>
  // CHECK:         p4hir.read %[[A]] : <!b5i>
  p4hir.func @assign_slice_of_field(%arg0 : !b5i, %arg1 : !b4i) -> !b5i {
    %s = p4hir.variable ["s", init] : <!S>
    %s_val = p4hir.struct (%arg0, %arg1) : !S
    p4hir.assign %s_val, %s : <!S>
    %a_ref = p4hir.struct_field_ref %s["a"] : <!S>
    p4hir.assign_slice %arg1, %a_ref[4 : 1] : !b4i -> <!b5i>
    %a = p4hir.read %a_ref : <!b5i>
    p4hir.return %a : !b5i
  }
}
