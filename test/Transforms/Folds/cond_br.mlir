// RUN: p4mlir-opt -allow-unregistered-dialect --canonicalize %s | FileCheck %s

/// Test the folding of CondBranchOp with a constant condition.

// CHECK-LABEL: func @cond_br_folding(
p4hir.func @cond_br_folding(%cond : !p4hir.bool, %a : !p4hir.int<32>) {
  // CHECK-NEXT: p4hir.return

  %false_cond = p4hir.const #p4hir.bool<false> : !p4hir.bool
  %true_cond = p4hir.const #p4hir.bool<true> : !p4hir.bool
  p4hir.cond_br %cond ^bb1, ^bb2(%a : !p4hir.int<32>)

^bb1:
  p4hir.cond_br %true_cond ^bb3, ^bb2(%a : !p4hir.int<32>)

^bb2(%x : !p4hir.int<32>):
  p4hir.cond_br %false_cond ^bb2(%x : !p4hir.int<32>), ^bb3

^bb3:
  p4hir.return
}

/// Test the folding of CondBranchOp when the successors are identical.

// CHECK-LABEL: func @cond_br_same_successor(
p4hir.func @cond_br_same_successor(%cond : !p4hir.bool, %a : !p4hir.int<32>) {
  // CHECK-NEXT: return

  p4hir.cond_br %cond ^bb1(%a : !p4hir.int<32>), ^bb1(%a : !p4hir.int<32>)

^bb1(%result : !p4hir.int<32>):
  p4hir.return
}


/// Test the compound folding of BranchOp and CondBranchOp.

// CHECK-LABEL: func @cond_br_and_br_folding(
p4hir.func @cond_br_and_br_folding(%a : !p4hir.int<32>) {
  // CHECK-NEXT: return

  %false_cond = p4hir.const #p4hir.bool<false> : !p4hir.bool
  %true_cond = p4hir.const #p4hir.bool<true> : !p4hir.bool
  p4hir.cond_br %true_cond ^bb2, ^bb1(%a : !p4hir.int<32>)

^bb1(%x : !p4hir.int<32>):
  p4hir.cond_br %false_cond ^bb1(%x : !p4hir.int<32>), ^bb2

^bb2:
  p4hir.return
}

/// Test the failure modes of collapsing CondBranchOp pass-throughs successors.

// CHECK-LABEL: func @cond_br_pass_through_fail(
p4hir.func @cond_br_pass_through_fail(%cond : !p4hir.bool) {
  // CHECK: p4hir.cond_br %{{.*}} ^bb1, ^bb2

  p4hir.cond_br %cond ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: "foo.op"
  // CHECK: p4hir.br ^bb2

  // Successors can't be collapsed if they contain other operations.
  "foo.op"() : () -> ()
  p4hir.br ^bb2

^bb2:
  p4hir.return
}

// CHECK-LABEL: @branchCondProp
//       CHECK:       %[[trueval:.+]] = p4hir.const #true
//       CHECK:       %[[falseval:.+]] = p4hir.const #false
//       CHECK:       "test.consumer1"(%[[trueval]]) : (!p4hir.bool) -> ()
//       CHECK:       "test.consumer2"(%[[falseval]]) : (!p4hir.bool) -> ()
p4hir.func @branchCondProp(%arg0: !p4hir.bool) {
  p4hir.cond_br %arg0 ^trueB, ^falseB

^trueB:
  "test.consumer1"(%arg0) : (!p4hir.bool) -> ()
  p4hir.br ^exit

^falseB:
  "test.consumer2"(%arg0) : (!p4hir.bool) -> ()
  p4hir.br ^exit

^exit:
  p4hir.return
}

// Make sure we terminate when other passes can't help us.

// CHECK-LABEL:   @unsimplified_cycle_2
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.bool) {
// CHECK:           p4hir.cond_br %[[ARG0]] ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           p4hir.br ^bb2 {A}
// CHECK:         ^bb2:
// CHECK:           p4hir.br ^bb2 {E}
// CHECK:         ^bb3:
// CHECK:           p4hir.br ^bb1
p4hir.func @unsimplified_cycle_2(%c : !p4hir.bool) {
  p4hir.cond_br %c ^bb6, ^bb7
^bb6:
  p4hir.br ^bb5 {F}
^bb5:
  p4hir.br ^bb1 {A}
^bb1:
  p4hir.br ^bb2 {B}
^bb2:
  p4hir.br ^bb3 {C}
^bb3:
  p4hir.br ^bb4 {D}
^bb4:
  p4hir.br ^bb1 {E}
^bb7:
  p4hir.br ^bb6
}

// CHECK-LABEL:   @unsimplified_cycle_1
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.bool) {
// CHECK:           p4hir.cond_br %[[ARG0]] ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           p4hir.br ^bb2
// CHECK:         ^bb2:
// CHECK:           p4hir.br ^bb3
// CHECK:         ^bb3:
// CHECK:           p4hir.br ^bb3
p4hir.func @unsimplified_cycle_1(%c : !p4hir.bool) {
  p4hir.cond_br %c ^bb1, ^bb2
^bb1:
  p4hir.br ^bb2
^bb2:
  p4hir.br ^bb3
^bb3:
  p4hir.br ^bb4
^bb4:
  p4hir.br ^bb3
}

