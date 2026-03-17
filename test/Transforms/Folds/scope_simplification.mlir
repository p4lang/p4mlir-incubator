// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!ref_i32i = !p4hir.ref<!i32i>
#int1_i32i = #p4hir.int<1> : !i32i
#int2_i32i = #p4hir.int<2> : !i32i
#int3_i32i = #p4hir.int<3> : !i32i

module {
  // CHECK-LABEL: @empty_scope_removed
  p4hir.func @empty_scope_removed(%arg0: !ref_i32i) {
    // Empty scope (just yield) should be inlined/removed
    // CHECK-NOT: p4hir.scope
    // CHECK: %[[C1:.*]] = p4hir.const #int1_i32i
    // CHECK-NEXT: p4hir.assign %[[C1]], %arg0
    // CHECK-NEXT: p4hir.return
    %c1 = p4hir.const #int1_i32i
    p4hir.scope {
    }
    p4hir.assign %c1, %arg0 : <!i32i>
    p4hir.return
  }

  // CHECK-LABEL: @nested_empty_scopes
  p4hir.func @nested_empty_scopes(%arg0: !ref_i32i) {
    // Nested empty scopes should all be removed
    // CHECK-NOT: p4hir.scope
    // CHECK: p4hir.return
    p4hir.scope {
      p4hir.scope {
        p4hir.scope {
        }
      }
    }
    p4hir.return
  }

  // CHECK-LABEL: @nested_scopes_inlined
  p4hir.func @nested_scopes_inlined(%arg0: !ref_i32i) {
    // CHECK-NOT: p4hir.scope
    // CHECK: p4hir.const
    // CHECK: p4hir.const
    // CHECK: p4hir.assign
    // CHECK: p4hir.assign
    // CHECK: p4hir.return
    p4hir.scope {
      %c1 = p4hir.const #int1_i32i
      p4hir.assign %c1, %arg0 : <!i32i>
      p4hir.scope {
        %c2 = p4hir.const #int2_i32i
        p4hir.assign %c2, %arg0 : <!i32i>
      }
    }
    p4hir.return
  }

  // CHECK-LABEL: @scope_with_annotations_preserved
  p4hir.func @scope_with_annotations_preserved(%arg0: !ref_i32i) {
    // CHECK: p4hir.scope annotations {name = "test"} {
    // CHECK-NEXT: p4hir.assign
    // CHECK-NEXT: }
    p4hir.scope annotations {name = "test"} {
      %c1 = p4hir.const #int1_i32i
      p4hir.assign %c1, %arg0 : <!i32i>
    }
    p4hir.return
  }

  // CHECK-LABEL: @empty_scope_with_annotations_preserved
  p4hir.func @empty_scope_with_annotations_preserved(%arg0: !ref_i32i) {
    // Empty scopes with annotations should be preserved
    // CHECK: p4hir.scope annotations {name = "test"} {
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return
    p4hir.scope annotations {name = "test"} {
    }
    p4hir.return
  }

  // CHECK-LABEL: @empty_scope_with_atomic_removed
  p4hir.func @empty_scope_with_atomic_removed(%arg0: !ref_i32i) {
    // Empty scopes with only @atomic annotation should be removed
    // CHECK-NOT: p4hir.scope
    // CHECK: p4hir.return
    p4hir.scope annotations {atomic} {
    }
    p4hir.return
  }

  // CHECK-LABEL: @empty_scope_with_atomic_and_name_preserved
  p4hir.func @empty_scope_with_atomic_and_name_preserved(%arg0: !ref_i32i) {
    // Empty scopes with @atomic plus other annotations should be preserved
    // CHECK: p4hir.scope annotations {atomic, name = "debug"} {
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return
    p4hir.scope annotations {atomic, name = "debug"} {
    }
    p4hir.return
  }

  // CHECK-LABEL: @scope_in_if_else_without_vars
  p4hir.func @scope_in_if_else_without_vars(%cond: !p4hir.bool, %arg0: !ref_i32i) {
    // Scope without variables/annotations inside if else branch should be inlined
    // CHECK: p4hir.if
    // CHECK-NOT: p4hir.scope
    // CHECK: p4hir.assign
    // CHECK: p4hir.return
    p4hir.if %cond {
    } else {
      p4hir.scope {
        %c1 = p4hir.const #int1_i32i
        p4hir.assign %c1, %arg0 : <!i32i>
      }
    }
    p4hir.return
  }
}
