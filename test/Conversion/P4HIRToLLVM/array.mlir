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
