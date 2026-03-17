// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
#false = #p4hir.bool<false> : !p4hir.bool
#int1_i32i = #p4hir.int<1> : !i32i
#int3_i32i = #p4hir.int<3> : !i32i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: f1
  p4hir.func @f1(%arg0: !p4hir.bool) {
    // CHECK-NEXT: p4hir.return
    p4hir.if %arg0 {
    }

    p4hir.if %arg0 {
    } else {
    }

    p4hir.return
  }

  // CHECK-LABEL: f2
  p4hir.func @f2(%arg0: !p4hir.bool, %arg1: !p4hir.ref<!p4hir.bool>) {
    // CHECK:      %[[NOT:.*]] = p4hir.unary(not, %arg0) : !p4hir.bool
    // CHECK-NEXT: p4hir.if %[[NOT]] {
    // CHECK-NEXT:   p4hir.assign %false, %arg1 : <!p4hir.bool>
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return

    p4hir.if %arg0 {
    } else {
      %false = p4hir.const #false
      p4hir.assign %false, %arg1 : <!p4hir.bool>
    }

    p4hir.return
  }

  // CHECK-LABEL: f3
  p4hir.func @f3(%arg0: !p4hir.bool, %arg1: !p4hir.ref<!i32i>) {
    %c1_i32i = p4hir.const #int1_i32i
    %c3_i32i = p4hir.const #int3_i32i

    // CHECK:      p4hir.if %arg0 {
    // CHECK-NEXT:   p4hir.assign %c1_i32i, %arg1 : <!i32i>
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   p4hir.assign %c3_i32i, %arg1 : <!i32i>
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return

    %not = p4hir.unary(not, %arg0) : !p4hir.bool
    p4hir.if %not {
      p4hir.assign %c3_i32i, %arg1 : <!i32i>
    } else {
      p4hir.assign %c1_i32i, %arg1 : <!i32i>
    }

    p4hir.return
  }

  // CHECK-LABEL: f4
  p4hir.func @f4(%arg0: !p4hir.bool) -> !i32i {
    %c1_i32i = p4hir.const #int1_i32i
    %c3_i32i = p4hir.const #int3_i32i

    // CHECK:      %[[RESULT:.*]] = p4hir.if %arg0 -> !i32i {
    // CHECK-NEXT:   p4hir.yield %c1_i32i : !i32i
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   p4hir.yield %c3_i32i : !i32i
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return %[[RESULT]] : !i32i

    %result = p4hir.if %arg0 -> !i32i {
      p4hir.yield %c1_i32i : !i32i
    } else {
      p4hir.yield %c3_i32i : !i32i
    }

    p4hir.return %result : !i32i
  }

  // CHECK-LABEL: constant_fold_with_annotation
  p4hir.func @constant_fold_with_annotation(%arg0: !p4hir.ref<!i32i>) {
    // Constant true with annotation should create scope with annotation preserved
    // CHECK:      %c1_i32i = p4hir.const #int1_i32i
    // CHECK-NEXT: p4hir.scope annotations {name = "fast_path"} {
    // CHECK-NEXT:   p4hir.assign %c1_i32i, %arg0 : <!i32i>
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return

    %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
    %c1_i32i = p4hir.const #int1_i32i

    p4hir.if %true annotations {name = "fast_path"} {
      p4hir.assign %c1_i32i, %arg0 : <!i32i>
    }

    p4hir.return
  }

  // CHECK-LABEL: constant_fold_filters_branch_hints
  p4hir.func @constant_fold_filters_branch_hints(%arg0: !p4hir.ref<!i32i>) {
    // Constant true with likely annotation should create scope without likely (branch hint doesn't apply to scope)
    // CHECK:      %c1_i32i = p4hir.const #int1_i32i
    // CHECK-NEXT: p4hir.scope annotations {name = "debug"} {
    // CHECK-NEXT:   p4hir.assign %c1_i32i, %arg0 : <!i32i>
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return

    %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
    %c1_i32i = p4hir.const #int1_i32i

    p4hir.if %true annotations {likely, name = "debug"} {
      p4hir.assign %c1_i32i, %arg0 : <!i32i>
    }

    p4hir.return
  }

  // CHECK-LABEL: constant_fold_only_branch_hints
  p4hir.func @constant_fold_only_branch_hints(%arg0: !p4hir.ref<!i32i>) {
    // Constant false with only unlikely annotation - should inline directly (no annotations to preserve)
    // CHECK:      %c3_i32i = p4hir.const #int3_i32i
    // CHECK-NEXT: p4hir.assign %c3_i32i, %arg0 : <!i32i>
    // CHECK-NEXT: p4hir.return
    // CHECK-NOT:  p4hir.scope

    %false = p4hir.const #p4hir.bool<false> : !p4hir.bool
    %c2_i32i = p4hir.const #int3_i32i

    p4hir.if %false {
      %dummy = p4hir.const #int1_i32i
      p4hir.assign %dummy, %arg0 : <!i32i>
    } else annotations {unlikely} {
      p4hir.assign %c2_i32i, %arg0 : <!i32i>
    }

    p4hir.return
  }

}
