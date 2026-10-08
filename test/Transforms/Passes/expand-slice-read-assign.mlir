// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --p4hir-expand-slice-read-assign | FileCheck %s

// A slice read becomes a read of the whole object and a slice of its value. A
// slice assignment becomes a read of the whole object, the replacement of the
// slice in its value, and an assignment of the result: the bits of the slice are
// cleared with a mask, and the assigned value is shifted into their place.

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!i16i = !p4hir.int<16>
#inout = #p4hir<dir inout>

// The masks keep the bits outside of the slices, named by their hex values.
// CHECK-DAG: #[[$MASK_FFFF00FF:.+]] = #p4hir.int<4294902015> : !b32i
// CHECK-DAG: #[[$MASK_00FFFFFF:.+]] = #p4hir.int<16777215> : !b32i
// CHECK-DAG: #[[$MASK_FF00:.+]] = #p4hir.int<65280> : !b16i
// CHECK-DAG: #[[$MASK_00FF:.+]] = #p4hir.int<255> : !b16i
// CHECK-DAG: #[[$MASK_F00F:.+]] = #p4hir.int<-4081> : !i16i

module {
  // CHECK-LABEL: p4hir.func @middle(
  // CHECK-SAME:      %[[V:.*]]: !b8i)
  // CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!b32i>
  // CHECK-NEXT:    %[[OLD:.*]] = p4hir.read %[[X]] : <!b32i>
  // CHECK-NEXT:    %[[CAST:.*]] = p4hir.cast(%[[V]] : !b8i) : !b32i
  // CHECK-NEXT:    %[[EIGHT:.*]] = p4hir.const #int8_b32i
  // CHECK-NEXT:    %[[SHL:.*]] = p4hir.shl(%[[CAST]], %[[EIGHT]] : !b32i) : !b32i
  // CHECK-NEXT:    %[[MASK:.*]] = p4hir.const #[[$MASK_FFFF00FF]]
  // CHECK-NEXT:    %[[AND:.*]] = p4hir.binop(and, %[[OLD]], %[[MASK]]) : !b32i
  // CHECK-NEXT:    %[[OR:.*]] = p4hir.binop(or, %[[AND]], %[[SHL]]) : !b32i
  // CHECK-NEXT:    p4hir.assign %[[OR]], %[[X]] : <!b32i>
  // CHECK-NEXT:    p4hir.return
  p4hir.func @middle(%v : !b8i) {
    %x = p4hir.variable ["x"] : <!b32i>
    p4hir.assign_slice %v, %x[15 : 8] : !b8i -> <!b32i>
    p4hir.return
  }

  // A slice at the low end needs no shift, and a slice covering the whole
  // object leaves no bits to keep: no read is needed.
  // CHECK-LABEL: p4hir.func @ends(
  // CHECK-SAME:      %[[V:.*]]: !b8i, %[[W:.*]]: !b16i)
  // CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!b16i>
  // CHECK-NEXT:    %[[OLD:.*]] = p4hir.read %[[X]] : <!b16i>
  // CHECK-NEXT:    %[[CAST:.*]] = p4hir.cast(%[[V]] : !b8i) : !b16i
  // CHECK-NEXT:    %[[MASK:.*]] = p4hir.const #[[$MASK_FF00]]
  // CHECK-NEXT:    %[[AND:.*]] = p4hir.binop(and, %[[OLD]], %[[MASK]]) : !b16i
  // CHECK-NEXT:    %[[OR:.*]] = p4hir.binop(or, %[[AND]], %[[CAST]]) : !b16i
  // CHECK-NEXT:    p4hir.assign %[[OR]], %[[X]] : <!b16i>
  // CHECK-NEXT:    %[[OLD2:.*]] = p4hir.read %[[X]] : <!b16i>
  // CHECK-NEXT:    %[[CAST2:.*]] = p4hir.cast(%[[V]] : !b8i) : !b16i
  // CHECK-NEXT:    %[[EIGHT:.*]] = p4hir.const #int8_b16i
  // CHECK-NEXT:    %[[SHL2:.*]] = p4hir.shl(%[[CAST2]], %[[EIGHT]] : !b16i) : !b16i
  // CHECK-NEXT:    %[[MASK2:.*]] = p4hir.const #[[$MASK_00FF]]
  // CHECK-NEXT:    %[[AND2:.*]] = p4hir.binop(and, %[[OLD2]], %[[MASK2]]) : !b16i
  // CHECK-NEXT:    %[[OR2:.*]] = p4hir.binop(or, %[[AND2]], %[[SHL2]]) : !b16i
  // CHECK-NEXT:    p4hir.assign %[[OR2]], %[[X]] : <!b16i>
  // CHECK-NEXT:    p4hir.assign %[[W]], %[[X]] : <!b16i>
  // CHECK-NEXT:    p4hir.return
  p4hir.func @ends(%v : !b8i, %w : !b16i) {
    %x = p4hir.variable ["x"] : <!b16i>
    p4hir.assign_slice %v, %x[7 : 0] : !b8i -> <!b16i>
    p4hir.assign_slice %v, %x[15 : 8] : !b8i -> <!b16i>
    p4hir.assign_slice %w, %x[15 : 0] : !b16i -> <!b16i>
    p4hir.return
  }

  // Slices are unsigned: the assigned value is zero-extended to the type of a
  // signed object.
  // CHECK-LABEL: p4hir.func @signed(
  // CHECK-SAME:      %[[V:.*]]: !b8i)
  // CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!i16i>
  // CHECK-NEXT:    %[[OLD:.*]] = p4hir.read %[[X]] : <!i16i>
  // CHECK-NEXT:    %[[CAST:.*]] = p4hir.cast(%[[V]] : !b8i) : !i16i
  // CHECK-NEXT:    %[[FOUR:.*]] = p4hir.const #int4_b16i
  // CHECK-NEXT:    %[[SHL:.*]] = p4hir.shl(%[[CAST]], %[[FOUR]] : !b16i) : !i16i
  // CHECK-NEXT:    %[[MASK:.*]] = p4hir.const #[[$MASK_F00F]]
  // CHECK-NEXT:    %[[AND:.*]] = p4hir.binop(and, %[[OLD]], %[[MASK]]) : !i16i
  // CHECK-NEXT:    %[[OR:.*]] = p4hir.binop(or, %[[AND]], %[[SHL]]) : !i16i
  // CHECK-NEXT:    p4hir.assign %[[OR]], %[[X]] : <!i16i>
  p4hir.func @signed(%v : !b8i) {
    %x = p4hir.variable ["x"] : <!i16i>
    p4hir.assign_slice %v, %x[11 : 4] : !b8i -> <!i16i>
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @read(
  // CHECK-SAME:      %[[X:[^:]*]]: !p4hir.ref<!i16i>
  // CHECK-NEXT:    %[[VAL:.*]] = p4hir.read %[[X]] : <!i16i>
  // CHECK-NEXT:    %[[Y:.*]] = p4hir.slice %[[VAL]][11 : 4] : !i16i -> !b8i
  // CHECK-NEXT:    p4hir.return %[[Y]] : !b8i
  p4hir.func @read(%x : !p4hir.ref<!i16i> {p4hir.dir = #inout, p4hir.param_name = "x"}) -> !b8i {
    %y = p4hir.read_slice %x[11 : 4] : <!i16i> -> !b8i
    p4hir.return %y : !b8i
  }

  // Any reference works, e.g. an `inout` parameter, and the expansion happens
  // where the slice access is, e.g. in a nested region.
  // CHECK-LABEL: p4hir.func action @param(
  // CHECK-SAME:      %[[X:[^:]*]]: !p4hir.ref<!b32i> {{.*}}, %[[C:.*]]: !p4hir.bool, %[[V:.*]]: !b8i)
  // CHECK-NEXT:    p4hir.if %[[C]] {
  // CHECK-NEXT:      %[[VAL:.*]] = p4hir.read %[[X]] : <!b32i>
  // CHECK-NEXT:      %[[LOW:.*]] = p4hir.slice %[[VAL]][7 : 0] : !b32i -> !b8i
  // CHECK-NEXT:      %[[OLD:.*]] = p4hir.read %[[X]] : <!b32i>
  // CHECK-NEXT:      %[[CAST:.*]] = p4hir.cast(%[[LOW]] : !b8i) : !b32i
  // CHECK-NEXT:      %[[SHIFT:.*]] = p4hir.const #int24_b32i
  // CHECK-NEXT:      %[[SHL:.*]] = p4hir.shl(%[[CAST]], %[[SHIFT]] : !b32i) : !b32i
  // CHECK-NEXT:      %[[MASK:.*]] = p4hir.const #[[$MASK_00FFFFFF]]
  // CHECK-NEXT:      %[[AND:.*]] = p4hir.binop(and, %[[OLD]], %[[MASK]]) : !b32i
  // CHECK-NEXT:      %[[OR:.*]] = p4hir.binop(or, %[[AND]], %[[SHL]]) : !b32i
  // CHECK-NEXT:      p4hir.assign %[[OR]], %[[X]] : <!b32i>
  // CHECK-NEXT:    }
  p4hir.func action @param(%x : !p4hir.ref<!b32i> {p4hir.dir = #inout, p4hir.param_name = "x"}, %c : !p4hir.bool, %v : !b8i) {
    p4hir.if %c {
      %low = p4hir.read_slice %x[7 : 0] : <!b32i> -> !b8i
      p4hir.assign_slice %low, %x[31 : 24] : !b8i -> <!b32i>
    }
    p4hir.return
  }
}
