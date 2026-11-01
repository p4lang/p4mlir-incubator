// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Structs lower to literal packed LLVM structs with the fields in the same order,
// nested structs included. The P4 name does not carry over.

// CHECK-LABEL: @nested(
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %{{.*}} = llvm.alloca %[[ONE]] x !llvm.struct<packed (i32, struct<packed (i8, i1)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ONE_E:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %{{.*}} = llvm.alloca %[[ONE_E]] x !llvm.struct<packed ()> : (i64) -> !llvm.ptr

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!inner = !p4hir.struct<"inner", x: !b8i, y: !p4hir.bool>
!outer = !p4hir.struct<"outer", a: !b32i, s: !inner>
!empty = !p4hir.struct<"empty">

module {
  p4hir.func @nested() {
    %s = p4hir.variable ["s", init] : <!outer>
    %e = p4hir.variable ["e", init] : <!empty>
    p4hir.return
  }
}

// -----

// Struct types are structural in LLVM: structs with the same field types are the
// same LLVM type whatever their names, while `p4hir-tuple-to-struct` gives the same
// name to structs with different field types.

// CHECK-LABEL: @same(
// CHECK:         llvm.alloca %{{.*}} x !llvm.struct<packed (i8, i8)>
// CHECK:         llvm.alloca %{{.*}} x !llvm.struct<packed (i8, i8)>
// CHECK-LABEL: @tuples(
// CHECK:         llvm.alloca %{{.*}} x !llvm.struct<packed (i8, i8)>
// CHECK:         llvm.alloca %{{.*}} x !llvm.struct<packed (i8, i1)>

!b8i = !p4hir.bit<8>
!a = !p4hir.struct<"a", x: !b8i, y: !b8i>
!b = !p4hir.struct<"b", u: !b8i, v: !b8i>
!t0 = !p4hir.struct<"_tupleToStruct", element_0: !b8i, element_1: !b8i>
!t1 = !p4hir.struct<"_tupleToStruct", element_0: !b8i, element_1: !p4hir.bool>

module {
  p4hir.func @same() {
    %a = p4hir.variable ["a", init] : <!a>
    %b = p4hir.variable ["b", init] : <!b>
    p4hir.return
  }

  p4hir.func @tuples() {
    %t0 = p4hir.variable ["t0", init] : <!t0>
    %t1 = p4hir.variable ["t1", init] : <!t1>
    p4hir.return
  }
}

// -----

// Struct values merging at a join become struct block arguments.

// CHECK-LABEL: @merge(
// CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %arg1 : !s to !llvm.struct<packed (i8, i1)>
// CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %arg2 : !s to !llvm.struct<packed (i8, i1)>
// CHECK:         llvm.cond_br %{{.*}}, ^[[T:bb[0-9]+]], ^[[F:bb[0-9]+]]
// CHECK:       ^[[T]]:
// CHECK-NEXT:    llvm.br ^[[JOIN:bb[0-9]+]](%[[A]] : !llvm.struct<packed (i8, i1)>)
// CHECK:       ^[[F]]:
// CHECK-NEXT:    llvm.br ^[[JOIN]](%[[B]] : !llvm.struct<packed (i8, i1)>)
// CHECK:       ^[[JOIN]](%{{.*}}: !llvm.struct<packed (i8, i1)>):

!b8i = !p4hir.bit<8>
!s = !p4hir.struct<"s", x: !b8i, y: !p4hir.bool>

module {
  p4hir.func @merge(%c : !p4hir.bool, %a : !s, %b : !s) -> !s {
    p4hir.cond_br %c ^t, ^f
  ^t:
    p4hir.br ^join(%a : !s)
  ^f:
    p4hir.br ^join(%b : !s)
  ^join(%r : !s):
    p4hir.return %r : !s
  }
}
