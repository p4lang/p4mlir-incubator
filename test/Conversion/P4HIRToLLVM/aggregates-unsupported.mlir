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

// -----

// `llvm.extractvalue` needs a constant position within the bounds of the array:
// reads at a dynamic index and at a constant one out of bounds, possibly negative,
// are left unconverted.

// CHECK-LABEL: @get(
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b32i
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b32i
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !i8i

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!i8i = !p4hir.int<8>
!arr = !p4hir.array<2 x !b8i>

module {
  p4hir.func @get(%arr : !arr, %i : !b32i) -> !b8i {
    %dyn = p4hir.array_get %arr[%i] : !arr, !b32i
    %c2 = p4hir.const #p4hir.int<2> : !b32i
    %oob = p4hir.array_get %arr[%c2] : !arr, !b32i
    %cm1 = p4hir.const #p4hir.int<-1> : !i8i
    %neg = p4hir.array_get %arr[%cm1] : !arr, !i8i
    %s0 = p4hir.binop(add, %dyn, %oob) : !b8i
    %s1 = p4hir.binop(add, %s0, %neg) : !b8i
    p4hir.return %s1 : !b8i
  }
}

// -----

// Building and reading headers, or structs containing them, is left unconverted.

// CHECK-LABEL: @header(
// CHECK:         p4hir.struct_extract %{{.*}}["h"] : !with_header
// CHECK:         p4hir.struct_extract %{{.*}}["a"] : !h
// CHECK:         p4hir.struct (%{{.*}}, %{{.*}}) : !h

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>

module {
  p4hir.func @header(%s : !with_header, %v : !validity_bit) -> !b8i {
    %h = p4hir.struct_extract %s["h"] : !with_header
    %a = p4hir.struct_extract %h["a"] : !h
    %n = p4hir.struct (%a, %v) : !h
    p4hir.return %a : !b8i
  }
}
