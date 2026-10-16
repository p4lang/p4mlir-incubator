// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s | p4mlir-opt | FileCheck %s

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @scoped
// CHECK:         %[[X:.*]] = p4hir.variable ["x"] : <!b8i>
// CHECK:         p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK:           p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NEXT:    }

module {
  p4hir.func @scoped() {
    %x = p4hir.variable ["x"] : <!b8i>
    p4hir.scope {
      p4hir.lifetime_start %x : <!b8i>
      %one = p4hir.const #p4hir.int<1> : !b8i
      p4hir.assign %one, %x : <!b8i>
      p4hir.lifetime_end %x : <!b8i>
    }
    p4hir.return
  }
}
