// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: p4hir.func @max(%arg0: !b16i {p4hir.dir = #in}, %arg1: !b16i {p4hir.dir = #in}) -> !b16i
// CHECK:    %[[CMP:.*]] = p4hir.cmp(gt, %arg0, %arg1) : !b16i, !p4hir.bool
// CHECK:    p4hir.if %[[CMP]] {
// CHECK:      p4hir.return %arg0 : !b16i
// CHECK:    }
// CHECK:    p4hir.return %arg1 : !b16i
// CHECK:  }

bit<16> max(in bit<16> left, in bit<16> right) {
    if (left > right)
        return left;
    return right;
}

// CHECK-LABEL: p4hir.func action @bar(%arg0: !b16i {p4hir.dir = #in}, %arg1: !b16i {p4hir.dir = #in}, %arg2: !p4hir.ref<!b16i> {p4hir.dir = #p4hir<dir out>}) {
// CHECK:    %[[CALL:.*]] = p4hir.call @max (%arg0, %arg1) : (!b16i, !b16i) -> !b16i
// CHECK:    p4hir.assign %[[CALL]], %arg2 : <!b16i>
// CHECK:    p4hir.return

action bar(in bit<16> arg1, in bit<16> arg2, out bit<16> res) {
  res = max(arg1, arg2);
}

// Example from P4 language spec (6.8. Calling convention: call by copy in/copy out)
// The function call is equivalent to the following sequence of statements:
// bit tmp1 = a;     // evaluate a; save result
// bit tmp2 = g(a);  // evaluate g(a); save result; modifies a
// f(tmp1, tmp2);    // evaluate f; modifies tmp1
// a = tmp1;         // copy inout result back into a
// However, we limit the scope of temporaries via structured control flow

// CHECK: p4hir.func @f(!p4hir.ref<!b1i> {p4hir.dir = #inout}, !b1i {p4hir.dir = #in})
extern void f(inout bit x, in bit y);
// CHECK: p4hir.func @g(!p4hir.ref<!b1i> {p4hir.dir = #inout}) -> !b1i
extern bit g(inout bit z);

action test_param() {
  bit a;
  f(a, g(a));
}

// CHECK-LABEL: p4hir.func action @test_param() {
// CHECK:    %[[A:.*]] = p4hir.variable ["a"] : <!b1i>
// CHECK:    p4hir.scope {
// CHECK:      %[[X_INOUT:.*]] = p4hir.variable ["x_inout_arg", init] : <!b1i>
// CHECK:      %[[A_VAL:.*]] = p4hir.read %[[A]] : <!b1i>
// CHECK:      p4hir.assign %[[A_VAL]], %[[X_INOUT]] : <!b1i>
// CHECK:      %[[G_VAL:.*]] = p4hir.scope {
// CHECK:        %[[Z_INOUT:.*]] = p4hir.variable ["z_inout_arg", init] : <!b1i>
// CHECK:        %[[A_VAL2:.*]] = p4hir.read %[[A]] : <!b1i>
// CHECK:        p4hir.assign %[[A_VAL2]], %[[Z_INOUT]] : <!b1i>
// CHECK:        %[[G_RES:.*]] = p4hir.call @g (%[[Z_INOUT]]) : (!p4hir.ref<!b1i>) -> !b1i
// CHECK:        %[[A_OUT_VAL:.*]] = p4hir.read %[[Z_INOUT]] : <!b1i>
// CHECK:        p4hir.assign %[[A_OUT_VAL]], %[[A]] : <!b1i>
// CHECK:        p4hir.yield %[[G_RES]] : !b1i
// CHECK:      } : !b1i
// CHECK:      p4hir.call @f (%[[X_INOUT]], %[[G_VAL]]) : (!p4hir.ref<!b1i>, !b1i) -> ()
// CHECK:      %[[A_OUT_VAL2:.*]] = p4hir.read %[[X_INOUT]] : <!b1i>
// CHECK:      p4hir.assign %[[A_OUT_VAL2]], %[[A]] : <!b1i>
// CHECK:    }
// CHECK:    p4hir.return
// CHECK:  }
