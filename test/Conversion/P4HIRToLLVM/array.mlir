// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Arrays lower to LLVM arrays, arrays of structs and structs of arrays included.
// P4 accepts arrays of size 0, and so does LLVM.

// CHECK-LABEL: @arrays(
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<4 x i8>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<2 x i1>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<2 x array<3 x i8>>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<0 x i8>

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!s = !p4hir.struct<"s", x: !b8i, a: !p4hir.array<2 x !b16i>>

module {
  p4hir.func @arrays() {
    %bits = p4hir.variable ["bits", init] : <!p4hir.array<4 x !b8i>>
    %bools = p4hir.variable ["bools", init] : <!p4hir.array<2 x !p4hir.bool>>
    %structs = p4hir.variable ["structs", init] : <!p4hir.array<3 x !s>>
    %nested = p4hir.variable ["nested", init] : <!p4hir.array<2 x !p4hir.array<3 x !b8i>>>
    %empty = p4hir.variable ["empty", init] : <!p4hir.array<0 x !b8i>>
    p4hir.return
  }
}

// -----

// An array is built by inserting its elements in order into a poison value, and an
// element at a constant index is read with `llvm.extractvalue`, whatever the width
// and signedness of the index.

// CHECK-LABEL: @get(
// CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !b8i to i8
// CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %arg1 : !b8i to i8
// CHECK:         %[[P:.*]] = llvm.mlir.poison : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A0:.*]] = llvm.insertvalue %[[A]], %[[P]][0] : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A1:.*]] = llvm.insertvalue %[[B]], %[[A0]][1] : !llvm.array<2 x i8>
// CHECK:         %[[E0:.*]] = llvm.extractvalue %[[A1]][1] : !llvm.array<2 x i8>
// CHECK:         %[[E1:.*]] = llvm.extractvalue %[[A1]][0] : !llvm.array<2 x i8>
// CHECK-NEXT:    llvm.add %[[E0]], %[[E1]] : i8

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!i8i = !p4hir.int<8>
!arr = !p4hir.array<2 x !b8i>

module {
  p4hir.func @get(%a : !b8i, %b : !b8i) -> !b8i {
    %arr = p4hir.array [%a, %b] : !arr
    %c1 = p4hir.const #p4hir.int<1> : !b32i
    %e0 = p4hir.array_get %arr[%c1] : !arr, !b32i
    %c0 = p4hir.const #p4hir.int<0> : !i8i
    %e1 = p4hir.array_get %arr[%c0] : !arr, !i8i
    %s = p4hir.binop(add, %e0, %e1) : !b8i
    p4hir.return %s : !b8i
  }
}

// -----

// Arrays of structs and structs of arrays compose.

// CHECK-LABEL: @nested(
// CHECK-NEXT:    %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !{{.*}} to !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[S:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK-NEXT:    %[[B:.*]] = llvm.extractvalue %[[S]][1] : !llvm.struct<packed (i8, array<2 x i16>)>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[E:.*]] = llvm.extractvalue %[[B]][1] : !llvm.array<2 x i16>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[E]] : i16 to !b16i

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i, a: !p4hir.array<2 x !b16i>>
!arr = !p4hir.array<3 x !s>

module {
  p4hir.func @nested(%arr : !arr) -> !b16i {
    %c2 = p4hir.const #p4hir.int<2> : !b32i
    %s = p4hir.array_get %arr[%c2] : !arr, !b32i
    %a = p4hir.struct_extract %s["a"] : !s
    %c1 = p4hir.const #p4hir.int<1> : !b32i
    %e = p4hir.array_get %a[%c1] : !p4hir.array<2 x !b16i>, !b32i
    p4hir.return %e : !b16i
  }
}
