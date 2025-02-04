// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: p4hir.func @max(%arg0: !p4hir.bit<16> {p4hir.dir = #p4hir<dir in>}, %arg1: !p4hir.bit<16> {p4hir.dir = #p4hir<dir in>}) -> !p4hir.bit<16>
// CHECK:    %[[CMP:.*]] = p4hir.cmp(gt, %arg0, %arg1) : !p4hir.bit<16>, !p4hir.bool
// CHECK:    p4hir.if %[[CMP]] {
// CHECK:      p4hir.return %arg0 : !p4hir.bit<16>
// CHECK:    }
// CHECK:    p4hir.return %arg1 : !p4hir.bit<16>
// CHECK:  }

bit<16> max(in bit<16> left, in bit<16> right) {
    if (left > right)
        return left;
    return right;
}

// CHECK-LABEL: p4hir.func action @bar(%arg0: !p4hir.bit<16> {p4hir.dir = #p4hir<dir in>}, %arg1: !p4hir.bit<16> {p4hir.dir = #p4hir<dir in>}, %arg2: !p4hir.ref<!p4hir.bit<16>> {p4hir.dir = #p4hir<dir out>}) {
// CHECK:    %[[CALL:.*]] = p4hir.call @max(%arg0, %arg1) : (!p4hir.bit<16>, !p4hir.bit<16>) -> !p4hir.bit<16>
// CHECK:    p4hir.store %[[CALL]], %arg2 : !p4hir.bit<16>, !p4hir.ref<!p4hir.bit<16>>
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

// CHECK: p4hir.func @f(!p4hir.ref<!p4hir.bit<1>> {p4hir.dir = #p4hir<dir inout>}, !p4hir.bit<1> {p4hir.dir = #p4hir<dir in>})
extern void f(inout bit x, in bit y);
// CHECK: p4hir.func @g(!p4hir.ref<!p4hir.bit<1>> {p4hir.dir = #p4hir<dir inout>}) -> !p4hir.bit<1>
extern bit g(inout bit z);

action test_param() {
  bit a;
  f(a, g(a));
}

// CHECK-LABEL: p4hir.func action @test_param() {
// CHECK:    %[[A:.*]] = p4hir.alloca !p4hir.bit<1> ["a"] : !p4hir.ref<!p4hir.bit<1>>
// CHECK:    p4hir.scope {
// CHECK:      %[[X_INOUT:.*]] = p4hir.alloca !p4hir.bit<1> ["x_inout", init] : !p4hir.ref<!p4hir.bit<1>>
// CHECK:      %[[A_VAL:.*]] = p4hir.load %[[A]] : !p4hir.ref<!p4hir.bit<1>>, !p4hir.bit<1>
// CHECK:      p4hir.store %[[A_VAL]], %[[X_INOUT]] : !p4hir.bit<1>, !p4hir.ref<!p4hir.bit<1>>
// CHECK:      %[[G_VAL:.*]] = p4hir.scope {
// CHECK:        %[[Z_INOUT:.*]] = p4hir.alloca !p4hir.bit<1> ["z_inout", init] : !p4hir.ref<!p4hir.bit<1>>
// CHECK:        %[[A_VAL2:.*]] = p4hir.load %[[A]] : !p4hir.ref<!p4hir.bit<1>>, !p4hir.bit<1>
// CHECK:        p4hir.store %[[A_VAL2]], %[[Z_INOUT]] : !p4hir.bit<1>, !p4hir.ref<!p4hir.bit<1>>
// CHECK:        %[[G_RES:.*]] = p4hir.call @g(%[[Z_INOUT]]) : (!p4hir.ref<!p4hir.bit<1>>) -> !p4hir.bit<1>
// CHECK:        %[[A_OUT_VAL:.*]] = p4hir.load %[[Z_INOUT]] : !p4hir.ref<!p4hir.bit<1>>, !p4hir.bit<1>
// CHECK:        p4hir.store %[[A_OUT_VAL]], %[[A]] : !p4hir.bit<1>, !p4hir.ref<!p4hir.bit<1>>
// CHECK:        p4hir.yield %[[G_RES]] : !p4hir.bit<1>
// CHECK:      } : !p4hir.bit<1>
// CHECK:      p4hir.call @f(%[[X_INOUT]], %[[G_VAL]]) : (!p4hir.ref<!p4hir.bit<1>>, !p4hir.bit<1>) -> ()
// CHECK:      %[[A_OUT_VAL2:.*]] = p4hir.load %[[X_INOUT]] : !p4hir.ref<!p4hir.bit<1>>, !p4hir.bit<1>
// CHECK:      p4hir.store %[[A_OUT_VAL2]], %[[A]] : !p4hir.bit<1>, !p4hir.ref<!p4hir.bit<1>>
// CHECK:    }
// CHECK:    p4hir.return
// CHECK:  }
