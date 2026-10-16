// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --p4hir-remove-soft-cf --canonicalize --p4hir-hoist-variables --p4hir-flatten-cfg | FileCheck %s

// Hoisting runs after soft control flow is removed and the IR is canonicalized,
// and before the CFG is flattened, while the scopes of variables are still
// regions. Their lifetime markers then delimit the blocks the scopes are
// flattened into, and every variable is allocated in the entry block.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @f(
// CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x_inout_arg", init] : <!b8i>
// CHECK-NEXT:    %[[Y:.*]] = p4hir.variable ["y_inout_arg", init] : <!b8i>
// CHECK-NEXT:    p4hir.cond_br
// CHECK-NOT:     p4hir.variable
// CHECK:         p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK-NEXT:    p4hir.assign %{{.*}}, %[[X]] : <!b8i>
// CHECK-NEXT:    p4hir.call @g (%[[X]])
// CHECK-NEXT:    p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NOT:     p4hir.variable
// CHECK:         p4hir.lifetime_start %[[Y]] : <!b8i>
// CHECK-NEXT:    p4hir.assign %{{.*}}, %[[Y]] : <!b8i>
// CHECK-NEXT:    p4hir.call @g (%[[Y]])
// CHECK-NEXT:    p4hir.lifetime_end %[[Y]] : <!b8i>
// CHECK-NOT:     p4hir.variable
// CHECK:         p4hir.return

module {
  p4hir.func @g(!p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"})

  p4hir.func @f(%cond : !p4hir.bool, %arg : !b8i) {
    p4hir.if %cond {
      p4hir.scope {
        %x = p4hir.variable ["x_inout_arg", init] : <!b8i>
        p4hir.assign %arg, %x : <!b8i>
        p4hir.call @g(%x) : (!p4hir.ref<!b8i>) -> ()
      }
      p4hir.soft_return
    }
    p4hir.scope {
      %y = p4hir.variable ["y_inout_arg", init] : <!b8i>
      p4hir.assign %arg, %y : <!b8i>
      p4hir.call @g(%y) : (!p4hir.ref<!b8i>) -> ()
    }
    p4hir.return
  }
}
