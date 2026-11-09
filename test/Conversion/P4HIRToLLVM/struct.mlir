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

// -----

// A struct is built by inserting its fields in order into a poison value, and a
// field is read with `llvm.extractvalue` at the same index.

// CHECK-LABEL: @swap(
// CHECK-NEXT:    %[[S:.*]] = builtin.unrealized_conversion_cast %arg0 : !s to !llvm.struct<packed (i8, i16)>
// CHECK-NEXT:    %[[X:.*]] = llvm.extractvalue %[[S]][0] : !llvm.struct<packed (i8, i16)>
// CHECK-NEXT:    %[[Y:.*]] = llvm.extractvalue %[[S]][1] : !llvm.struct<packed (i8, i16)>
// CHECK-NEXT:    %[[P:.*]] = llvm.mlir.poison : !llvm.struct<packed (i16, i8)>
// CHECK-NEXT:    %[[T0:.*]] = llvm.insertvalue %[[Y]], %[[P]][0] : !llvm.struct<packed (i16, i8)>
// CHECK-NEXT:    %[[T1:.*]] = llvm.insertvalue %[[X]], %[[T0]][1] : !llvm.struct<packed (i16, i8)>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[T1]] : !llvm.struct<packed (i16, i8)> to !t

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!s = !p4hir.struct<"s", x: !b8i, y: !b16i>
!t = !p4hir.struct<"t", y: !b16i, x: !b8i>

module {
  p4hir.func @swap(%s : !s) -> !t {
    %x = p4hir.struct_extract %s["x"] : !s
    %y = p4hir.struct_extract %s["y"] : !s
    %t = p4hir.struct (%y, %x) : !t
    p4hir.return %t : !t
  }
}

// -----

// Nested structs are built and read one level at a time. Tuples, which are structs
// after `p4hir-tuple-to-struct`, are no different.

// CHECK-LABEL: @nested(
// CHECK-DAG:     %[[O:.*]] = builtin.unrealized_conversion_cast %arg0 : !outer to !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-DAG:     %[[V:.*]] = builtin.unrealized_conversion_cast %arg1 : !b8i to i8
// CHECK:         %[[I:.*]] = llvm.extractvalue %[[O]][1] : !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-NEXT:    %[[Y:.*]] = llvm.extractvalue %[[I]][1] : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[P:.*]] = llvm.mlir.poison : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[T0:.*]] = llvm.insertvalue %[[V]], %[[P]][0] : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[T1:.*]] = llvm.insertvalue %[[Y]], %[[T0]][1] : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[T1]] : !llvm.struct<packed (i8, i1)> to !_tupleToStruct

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!inner = !p4hir.struct<"inner", x: !b8i, y: !p4hir.bool>
!outer = !p4hir.struct<"outer", a: !b32i, s: !inner>
!tuple = !p4hir.struct<"_tupleToStruct", element_0: !b8i, element_1: !p4hir.bool>

module {
  p4hir.func @nested(%o : !outer, %v : !b8i) -> !tuple {
    %inner = p4hir.struct_extract %o["s"] : !outer
    %y = p4hir.struct_extract %inner["y"] : !inner
    %t = p4hir.struct (%v, %y) : !tuple
    p4hir.return %t : !tuple
  }
}

// -----

// An empty struct has no fields to insert.

// CHECK-LABEL: @empty(
// CHECK-NEXT:    %[[P:.*]] = llvm.mlir.poison : !llvm.struct<packed ()>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[P]] : !llvm.struct<packed ()> to !empty

!empty = !p4hir.struct<"empty">

module {
  p4hir.func @empty() -> !empty {
    %e = p4hir.struct () : !empty
    p4hir.return %e : !empty
  }
}

// -----

// Struct constants are built from their fields like any other struct, which LLVM
// folds back into a constant. All-zero ones, e.g. default values, are a single
// `llvm.mlir.zero`, nested or not.

// CHECK-LABEL: @constants(
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK-NEXT:    %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %[[P:.*]] = llvm.mlir.poison : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[S0:.*]] = llvm.insertvalue %[[ONE]], %[[P]][0] : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[S1:.*]] = llvm.insertvalue %[[TRUE]], %[[S0]][1] : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[Z:.*]] = llvm.mlir.zero : !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[P2:.*]] = llvm.mlir.poison : !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-NEXT:    %[[N0:.*]] = llvm.insertvalue %[[TWO]], %[[P2]][0] : !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-NEXT:    %[[N1:.*]] = llvm.insertvalue %[[Z]], %[[N0]][1] : !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.zero : !llvm.struct<packed (i32, struct<packed (i8, i1)>)>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.zero : !llvm.struct<packed ()>

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!inner = !p4hir.struct<"inner", x: !b8i, y: !p4hir.bool>
!outer = !p4hir.struct<"outer", a: !b32i, s: !inner>
!empty = !p4hir.struct<"empty">
#int0_b8i = #p4hir.int<0> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int0_b32i = #p4hir.int<0> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool

module {
  p4hir.func @constants() {
    %flat = p4hir.const #p4hir.aggregate<[#int1_b8i, #true]> : !inner
    %nested = p4hir.const #p4hir.aggregate<[#int2_b32i, #p4hir.aggregate<[#int0_b8i, #false]> : !inner]> : !outer
    %zero = p4hir.const #p4hir.aggregate<[#int0_b32i, #p4hir.aggregate<[#int0_b8i, #false]> : !inner]> : !outer
    %empty = p4hir.const #p4hir.aggregate<[]> : !empty
    p4hir.return
  }
}
