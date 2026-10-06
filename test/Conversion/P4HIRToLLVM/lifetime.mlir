// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s
// RUN: p4mlir-opt %s --lower-p4hir-to-llvm=initialize-variables=false -split-input-file \
// RUN:   | FileCheck %s --check-prefix=NOINIT

// Lifetime markers lower to the LLVM intrinsics. A variable comes into
// existence holding the default value of its type: that value is stored where
// its lifetime starts, not where it is allocated, unless `initialize-variables`
// is off.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: @marked(
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[X:.*]] = llvm.alloca %[[ONE]] x i8 : (i64) -> !llvm.ptr
// CHECK-NOT:     llvm.store
// CHECK:       ^bb1:
// CHECK-NEXT:    llvm.intr.lifetime.start %[[X]] : !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:    llvm.store %[[ZERO]], %[[X]] : i8, !llvm.ptr
// CHECK-NEXT:    %{{.*}} = llvm.load %[[X]] : !llvm.ptr -> i8
// CHECK-NEXT:    llvm.intr.lifetime.end %[[X]] : !llvm.ptr

// NOINIT-LABEL: @marked(
// NOINIT-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// NOINIT-NEXT:    %[[X:.*]] = llvm.alloca %[[ONE]] x i8 : (i64) -> !llvm.ptr
// NOINIT-NOT:     llvm.store
// NOINIT:       ^bb1:
// NOINIT-NEXT:    llvm.intr.lifetime.start %[[X]] : !llvm.ptr
// NOINIT-NEXT:    %{{.*}} = llvm.load %[[X]] : !llvm.ptr -> i8
// NOINIT-NEXT:    llvm.intr.lifetime.end %[[X]] : !llvm.ptr

module {
  p4hir.func @marked() {
    %x = p4hir.variable ["x"] : <!b8i>
    p4hir.br ^bb1
  ^bb1:
    p4hir.lifetime_start %x : <!b8i>
    %v = p4hir.read %x : <!b8i>
    p4hir.lifetime_end %x : <!b8i>
    p4hir.return
  }
}

// -----

// A variable with `init` is initialized by its first use: no default value is
// stored.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: @initialized(
// CHECK-NEXT:    %[[ARG:.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[X:.*]] = llvm.alloca %[[ONE]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      llvm.intr.lifetime.start %[[X]] : !llvm.ptr
// CHECK-NEXT:      llvm.store %[[ARG]], %[[X]] : i8, !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.end %[[X]] : !llvm.ptr
// CHECK-NEXT:    }

module {
  p4hir.func @initialized() {
    %arg = p4hir.const #p4hir.int<7> : !b8i
    %x = p4hir.variable ["x", init] : <!b8i>
    p4hir.scope {
      p4hir.lifetime_start %x : <!b8i>
      p4hir.assign %arg, %x : <!b8i>
      p4hir.lifetime_end %x : <!b8i>
    }
    p4hir.return
  }
}

// -----

// LLVM only accepts lifetime markers on allocas: the markers of variables that
// are not lowered stay, e.g. those of `bit<0>` variables or of controls.

!b0i = !p4hir.bit<0>
!b8i = !p4hir.bit<8>

// CHECK-LABEL: @zero(
// CHECK-NEXT:    %[[Z:.*]] = p4hir.variable ["z"] : <!b0i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[Z]] : <!b0i>
// CHECK-NEXT:      p4hir.lifetime_end %[[Z]] : <!b0i>
// CHECK-NEXT:    }
// CHECK:       p4hir.control_apply {
// CHECK-NEXT:    %[[W:.*]] = p4hir.variable ["w"] : <!b8i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[W]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[W]] : <!b8i>
// CHECK-NEXT:    }

module {
  p4hir.func @zero() {
    %z = p4hir.variable ["z"] : <!b0i>
    p4hir.scope {
      p4hir.lifetime_start %z : <!b0i>
      p4hir.lifetime_end %z : <!b0i>
    }
    p4hir.return
  }

  p4hir.control @c()() {
    p4hir.control_apply {
      %w = p4hir.variable ["w"] : <!b8i>
      p4hir.scope {
        p4hir.lifetime_start %w : <!b8i>
        p4hir.lifetime_end %w : <!b8i>
      }
    }
  }
}
