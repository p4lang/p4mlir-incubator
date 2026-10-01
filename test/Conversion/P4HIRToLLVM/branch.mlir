// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Branch lowering converts successor block arguments. Entry-block conversion
// is left to the operation that owns the region.

// CHECK-LABEL: @select(
// CHECK:         %[[LT:.*]] = llvm.icmp "ult"
// CHECK-NEXT:    llvm.cond_br %[[LT]], ^[[T:bb[0-9]+]], ^[[F:bb[0-9]+]]
// CHECK-NEXT:  ^[[T]]:
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    llvm.br ^[[JOIN:bb[0-9]+]](%[[ONE]] : i32)
// CHECK-NEXT:  ^[[F]]:
// CHECK-NEXT:    %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    llvm.br ^[[JOIN]](%[[TWO]] : i32)
// CHECK-NEXT:  ^[[JOIN]](%{{.*}}: i32):

!b32i = !p4hir.bit<32>

module {
  p4hir.func @select(%a : !b32i, %b : !b32i) -> !b32i {
    %lt = p4hir.cmp(lt, %a : !b32i, %b : !b32i)
    p4hir.cond_br %lt ^t, ^f
  ^t:
    %one = p4hir.const #p4hir.int<1> : !b32i
    p4hir.br ^join(%one : !b32i)
  ^f:
    %two = p4hir.const #p4hir.int<2> : !b32i
    p4hir.br ^join(%two : !b32i)
  ^join(%r : !b32i):
    p4hir.return %r : !b32i
  }
}

// -----

// Operands are passed to both successors of a conditional branch. The loop
// header is converted once, by whichever branch to it is lowered first.

// CHECK-LABEL: @count(
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.br ^[[HEAD:bb[0-9]+]](%[[ZERO]] : i32)
// CHECK-NEXT:  ^[[HEAD]](%[[I:.*]]: i32):
// CHECK-NEXT:    %[[LT:.*]] = llvm.icmp "ult" %[[I]], %{{.*}} : i32
// CHECK-NEXT:    llvm.cond_br %[[LT]], ^[[BODY:bb[0-9]+]](%[[I]] : i32), ^[[EXIT:bb[0-9]+]](%[[I]] : i32)
// CHECK-NEXT:  ^[[BODY]](%[[J:.*]]: i32):
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[NEXT:.*]] = llvm.add %[[J]], %[[ONE]] : i32
// CHECK-NEXT:    llvm.br ^[[HEAD]](%[[NEXT]] : i32)
// CHECK-NEXT:  ^[[EXIT]](%{{.*}}: i32):

!b32i = !p4hir.bit<32>

module {
  p4hir.func @count(%n : !b32i) -> !b32i {
    %zero = p4hir.const #p4hir.int<0> : !b32i
    p4hir.br ^head(%zero : !b32i)
  ^head(%i : !b32i):
    %lt = p4hir.cmp(lt, %i : !b32i, %n : !b32i)
    p4hir.cond_br %lt ^body(%i : !b32i), ^exit(%i : !b32i)
  ^body(%j : !b32i):
    %one = p4hir.const #p4hir.int<1> : !b32i
    %next = p4hir.binop(add, %j, %one) : !b32i
    p4hir.br ^head(%next : !b32i)
  ^exit(%r : !b32i):
    p4hir.return %r : !b32i
  }
}

// -----

// Boolean block arguments lower to i1.

// CHECK-LABEL: @bool_arg(
// CHECK:         %[[EQ:.*]] = llvm.icmp "eq"
// CHECK-NEXT:    llvm.br ^[[COND:bb[0-9]+]](%[[EQ]] : i1)
// CHECK-NEXT:  ^[[COND]](%[[C:.*]]: i1):
// CHECK-NEXT:    llvm.cond_br %[[C]], ^{{bb[0-9]+}}, ^{{bb[0-9]+}}

!b32i = !p4hir.bit<32>

module {
  p4hir.func @bool_arg(%a : !b32i, %b : !b32i) {
    %eq = p4hir.cmp(eq, %a : !b32i, %b : !b32i)
    p4hir.br ^cond(%eq : !p4hir.bool)
  ^cond(%c : !p4hir.bool):
    p4hir.cond_br %c ^t, ^f
  ^t:
    p4hir.return
  ^f:
    p4hir.return
  }
}

// -----

// Distinct edge values must stay associated with their respective successors.

// CHECK-LABEL: @distinct_edges(
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    llvm.cond_br %{{.*}}, ^[[T:bb[0-9]+]](%[[ONE]] : i32), ^[[F:bb[0-9]+]](%[[TWO]] : i32)
// CHECK-NEXT:  ^[[T]](%{{.*}}: i32):
// CHECK-NEXT:    p4hir.return
// CHECK-NEXT:  ^[[F]](%{{.*}}: i32):
// CHECK-NEXT:    p4hir.return

module {
  p4hir.func @distinct_edges(%c: !p4hir.bool) {
    %one = p4hir.const #p4hir.int<1> : !p4hir.bit<32>
    %two = p4hir.const #p4hir.int<2> : !p4hir.bit<32>
    p4hir.cond_br %c ^t(%one : !p4hir.bit<32>), ^f(%two : !p4hir.bit<32>)
  ^t(%a: !p4hir.bit<32>):
    p4hir.return
  ^f(%b: !p4hir.bit<32>):
    p4hir.return
  }
}

// -----

// Both edges share a destination. Its signature is converted once, while each
// edge retains its own incoming value.

// CHECK-LABEL: @same_dest(
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    llvm.cond_br %{{.*}}, ^[[JOIN:bb[0-9]+]](%[[ONE]] : i32), ^[[JOIN]](%[[TWO]] : i32)
// CHECK-NEXT:  ^[[JOIN]](%{{.*}}: i32):
// CHECK-NEXT:    p4hir.return

module {
  p4hir.func @same_dest(%c: !p4hir.bool) {
    %one = p4hir.const #p4hir.int<1> : !p4hir.bit<32>
    %two = p4hir.const #p4hir.int<2> : !p4hir.bit<32>
    p4hir.cond_br %c ^join(%one : !p4hir.bit<32>), ^join(%two : !p4hir.bit<32>)
  ^join(%r: !p4hir.bit<32>):
    p4hir.return
  }
}

// -----

// Convert every argument in a mixed-type signature, preserving operand order
// through both unconditional and conditional branches.

// CHECK-LABEL: @mixed_args(
// CHECK:         %[[YES:.*]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %[[SMALL:.*]] = llvm.mlir.constant(8 : i8) : i8
// CHECK-NEXT:    %[[WIDE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK-NEXT:    llvm.br ^[[FORWARD:bb[0-9]+]](%[[YES]], %[[SMALL]], %[[WIDE]] : i1, i8, i32)
// CHECK-NEXT:  ^[[FORWARD]](%[[FLAG:.*]]: i1, %[[A:.*]]: i8, %[[B:.*]]: i32):
// CHECK-NEXT:    llvm.cond_br %{{.*}}, ^[[T:bb[0-9]+]](%[[FLAG]], %[[A]], %[[B]] : i1, i8, i32), ^[[F:bb[0-9]+]](%[[B]], %[[FLAG]], %[[A]] : i32, i1, i8)
// CHECK-NEXT:  ^[[T]](%{{.*}}: i1, %{{.*}}: i8, %{{.*}}: i32):
// CHECK-NEXT:    p4hir.return
// CHECK-NEXT:  ^[[F]](%{{.*}}: i32, %{{.*}}: i1, %{{.*}}: i8):
// CHECK-NEXT:    p4hir.return

module {
  p4hir.func @mixed_args(%c: !p4hir.bool) {
    %yes = p4hir.const #p4hir.bool<true> : !p4hir.bool
    %small = p4hir.const #p4hir.int<8> : !p4hir.bit<8>
    %wide = p4hir.const #p4hir.int<32> : !p4hir.bit<32>
    p4hir.br ^forward(%yes, %small, %wide : !p4hir.bool, !p4hir.bit<8>, !p4hir.bit<32>)
  ^forward(%flag: !p4hir.bool, %a: !p4hir.bit<8>, %b: !p4hir.bit<32>):
    p4hir.cond_br %c ^t(%flag, %a, %b : !p4hir.bool, !p4hir.bit<8>, !p4hir.bit<32>), ^f(%b, %flag, %a : !p4hir.bit<32>, !p4hir.bool, !p4hir.bit<8>)
  ^t(%tf: !p4hir.bool, %ta: !p4hir.bit<8>, %tb: !p4hir.bit<32>):
    p4hir.return
  ^f(%fb: !p4hir.bit<32>, %ff: !p4hir.bool, %fa: !p4hir.bit<8>):
    p4hir.return
  }
}

// -----

// Unsupported block arguments must leave the branch and successor consistent.

// CHECK-LABEL: @unsupported_br(
// CHECK-SAME: %[[X:.*]]: !infint)
// CHECK-NEXT:    p4hir.br ^[[DEST:bb[0-9]+]](%[[X]] : !infint)
// CHECK-NEXT:  ^[[DEST]](%{{.*}}: !infint):
// CHECK-NEXT:    p4hir.return
// CHECK-NEXT:  }

module {
  p4hir.func @unsupported_br(%x: !p4hir.infint) {
    p4hir.br ^dest(%x : !p4hir.infint)
  ^dest(%r: !p4hir.infint):
    p4hir.return
  }
}

// -----

// Failure to convert one successor must not leave a partially converted
// conditional branch or change the supported successor's signature.

// CHECK-LABEL: @unsupported_cond(
// CHECK-SAME: %[[C:.*]]: !p4hir.bool, %[[X:.*]]: !b32i, %[[Y:.*]]: !infint)
// CHECK-NEXT:    p4hir.cond_br %[[C]] ^[[T:bb[0-9]+]](%[[X]] : !b32i), ^[[F:bb[0-9]+]](%[[Y]] : !infint)
// CHECK-NEXT:  ^[[T]](%{{.*}}: !b32i):
// CHECK-NEXT:    p4hir.return
// CHECK-NEXT:  ^[[F]](%{{.*}}: !infint):
// CHECK-NEXT:    p4hir.return
// CHECK-NEXT:  }

module {
  p4hir.func @unsupported_cond(%c: !p4hir.bool, %x: !p4hir.bit<32>, %y: !p4hir.infint) {
    p4hir.cond_br %c ^t(%x : !p4hir.bit<32>), ^f(%y : !p4hir.infint)
  ^t(%a: !p4hir.bit<32>):
    p4hir.return
  ^f(%b: !p4hir.infint):
    p4hir.return
  }
}
