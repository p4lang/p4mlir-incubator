// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s
// RUN: p4mlir-opt %s --lower-p4hir-to-llvm=initialize-variables=false -split-input-file \
// RUN:   | FileCheck %s --check-prefix=NOINIT

// References lower to opaque pointers: variables to allocas, assignments to
// stores and reads to loads, which carry the object type.

!b32i = !p4hir.bit<32>
!i8i = !p4hir.int<8>

// CHECK-LABEL: @locals(
// CHECK-NEXT:    %[[C42:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT:    %[[ONE_A:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[A:.*]] = llvm.alloca %[[ONE_A]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[C42]], %[[A]] : i32, !llvm.ptr
// CHECK-NEXT:    %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %[[ONE_B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[B:.*]] = llvm.alloca %[[ONE_B]] x i1 : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[TRUE]], %[[B]] : i1, !llvm.ptr
// CHECK-NEXT:    %[[ONE_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[C:.*]] = llvm.alloca %[[ONE_C]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:    llvm.store %[[ZERO]], %[[C]] : i8, !llvm.ptr
// CHECK-NEXT:    %[[VAL:.*]] = llvm.load %[[A]] : !llvm.ptr -> i32
// CHECK-NEXT:    %{{.*}} = llvm.add %[[VAL]], %[[VAL]] : i32

module {
  p4hir.func @locals() {
    %c42 = p4hir.const #p4hir.int<42> : !b32i
    %a = p4hir.variable ["a", init] : <!b32i>
    p4hir.assign %c42, %a : <!b32i>
    %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
    %b = p4hir.variable ["b", init] : <!p4hir.bool>
    p4hir.assign %true, %b : <!p4hir.bool>
    %c = p4hir.variable ["c"] : <!i8i>
    %val = p4hir.read %a : <!b32i>
    %sum = p4hir.binop(add, %val, %val) : !b32i
    p4hir.return
  }
}

// -----

// P4 leaves the value of an uninitialized variable unspecified. As mem2reg does
// on P4HIR, the lowering takes the default value of its type, unless
// `initialize-variables` is off.

!b16i = !p4hir.bit<16>
!i16i = !p4hir.int<16>

// CHECK-LABEL: @defaults(
// CHECK-NEXT:    %[[ONE_U:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[U:.*]] = llvm.alloca %[[ONE_U]] x i16 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ZERO_U:.*]] = llvm.mlir.constant(0 : i16) : i16
// CHECK-NEXT:    llvm.store %[[ZERO_U]], %[[U]] : i16, !llvm.ptr
// CHECK-NEXT:    %[[ONE_S:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[S:.*]] = llvm.alloca %[[ONE_S]] x i16 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ZERO_S:.*]] = llvm.mlir.constant(0 : i16) : i16
// CHECK-NEXT:    llvm.store %[[ZERO_S]], %[[S]] : i16, !llvm.ptr
// CHECK-NEXT:    %[[ONE_B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[B:.*]] = llvm.alloca %[[ONE_B]] x i1 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT:    llvm.store %[[FALSE]], %[[B]] : i1, !llvm.ptr

// NOINIT-LABEL: @defaults(
// NOINIT-NEXT:    %[[ONE_U:.*]] = llvm.mlir.constant(1 : i64) : i64
// NOINIT-NEXT:    %{{.*}} = llvm.alloca %[[ONE_U]] x i16 : (i64) -> !llvm.ptr
// NOINIT-NEXT:    %[[ONE_S:.*]] = llvm.mlir.constant(1 : i64) : i64
// NOINIT-NEXT:    %{{.*}} = llvm.alloca %[[ONE_S]] x i16 : (i64) -> !llvm.ptr
// NOINIT-NEXT:    %[[ONE_B:.*]] = llvm.mlir.constant(1 : i64) : i64
// NOINIT-NEXT:    %{{.*}} = llvm.alloca %[[ONE_B]] x i1 : (i64) -> !llvm.ptr
// NOINIT-NEXT:    {{p4hir|llvm}}.return

module {
  p4hir.func @defaults() {
    %u = p4hir.variable ["u"] : <!b16i>
    %s = p4hir.variable ["s"] : <!i16i>
    %b = p4hir.variable ["b"] : <!p4hir.bool>
    p4hir.return
  }
}

// -----

// Variables declared in nested regions or in other blocks are allocated in place,
// not in the entry block.

!b32i = !p4hir.bit<32>

// CHECK-LABEL: @in_place(
// CHECK-NEXT:    %[[X:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      %[[ONE_A:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[A:.*]] = llvm.alloca %[[ONE_A]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:      llvm.store %[[X]], %[[A]] : i32, !llvm.ptr
// CHECK-NEXT:    }
// CHECK:       ^bb1:
// CHECK-NEXT:    %[[ONE_B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[B:.*]] = llvm.alloca %[[ONE_B]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[X]], %[[B]] : i32, !llvm.ptr

module {
  p4hir.func @in_place() {
    %x = p4hir.const #p4hir.int<7> : !b32i
    p4hir.scope {
      %a = p4hir.variable ["a", init] : <!b32i>
      p4hir.assign %x, %a : <!b32i>
    }
    p4hir.br ^bb1
  ^bb1:
    %b = p4hir.variable ["b", init] : <!b32i>
    p4hir.assign %x, %b : <!b32i>
    p4hir.return
  }
}

// -----

// A reference to an object without an LLVM counterpart, e.g. `bit<0>`, has no
// LLVM counterpart either: neither the variable nor its accesses are lowered.

!b0i = !p4hir.bit<0>

// CHECK-LABEL: @zero_var(
// CHECK-NEXT:    %[[ZERO:.*]] = p4hir.const
// CHECK-NEXT:    %[[VAR:.*]] = p4hir.variable ["var", init] : <!b0i>
// CHECK-NEXT:    p4hir.assign %[[ZERO]], %[[VAR]] : <!b0i>
// CHECK-NEXT:    %{{.*}} = p4hir.read %[[VAR]] : <!b0i>

module {
  p4hir.func @zero_var() {
    %zero = p4hir.const #p4hir.int<0> : !b0i
    %var = p4hir.variable ["var", init] : <!b0i>
    p4hir.assign %zero, %var : <!b0i>
    %val = p4hir.read %var : <!b0i>
    p4hir.return
  }
}

// -----

// Only variables of functions are stack slots, including those of control-local
// actions. Variables of controls and parsers are left to the lowering of their
// parents.

!b32i = !p4hir.bit<32>

// CHECK-LABEL: p4hir.control @c()() {
// CHECK:         p4hir.func action @local(
// CHECK:           llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr
// CHECK:         p4hir.control_apply {
// CHECK-NEXT:      p4hir.variable ["w"] : <!b32i>
// CHECK:       p4hir.parser @p()() {
// CHECK:         p4hir.state @start {
// CHECK-NEXT:      p4hir.variable ["s"] : <!b32i>

module {
  p4hir.control @c()() {
    p4hir.func action @local() {
      %x = p4hir.const #p4hir.int<7> : !b32i
      %v = p4hir.variable ["v", init] : <!b32i>
      p4hir.assign %x, %v : <!b32i>
      p4hir.return
    }
    p4hir.control_apply {
      %w = p4hir.variable ["w"] : <!b32i>
    }
  }

  p4hir.parser @p()() {
    p4hir.state @start {
      %s = p4hir.variable ["s"] : <!b32i>
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }
}
