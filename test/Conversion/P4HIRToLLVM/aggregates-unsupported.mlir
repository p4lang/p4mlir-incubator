// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Headers, header unions and header stacks are not lowered yet, and neither are
// structs and arrays containing them or other types without an LLVM counterpart:
// variables of these types are left unconverted.

!b8i = !p4hir.bit<8>
!infint = !p4hir.infint
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!u = !p4hir.header_union<"u", h: !h>
!hs = !p4hir.header_stack<2x!h>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>
!with_infint = !p4hir.struct<"with_infint", x: !b8i, i: !infint>

// CHECK-LABEL: @types(
// CHECK-NEXT:    p4hir.variable ["header", init] : <!h>
// CHECK-NEXT:    p4hir.variable ["header_union", init] : <!u>
// CHECK-NEXT:    p4hir.variable ["header_stack", init] : <!{{.*}}>
// CHECK-NEXT:    p4hir.variable ["struct_with_header", init] : <!with_header>
// CHECK-NEXT:    p4hir.variable ["struct_with_infint", init] : <!with_infint>
// CHECK-NEXT:    p4hir.variable ["array_of_infint", init] : <!{{.*}}>
// CHECK-NEXT:    p4hir.variable ["array_of_structs_with_header", init] : <!{{.*}}>

module {
  p4hir.func @types() {
    %header = p4hir.variable ["header", init] : <!h>
    %header_union = p4hir.variable ["header_union", init] : <!u>
    %header_stack = p4hir.variable ["header_stack", init] : <!hs>
    %struct_with_header = p4hir.variable ["struct_with_header", init] : <!with_header>
    %struct_with_infint = p4hir.variable ["struct_with_infint", init] : <!with_infint>
    %array_of_infint = p4hir.variable ["array_of_infint", init] : <!p4hir.array<2 x !infint>>
    %array_of_structs_with_header = p4hir.variable ["array_of_structs_with_header", init] : <!p4hir.array<2 x !with_header>>
    p4hir.return
  }
}
