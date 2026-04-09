// RUN: p4mlir-opt -allow-unregistered-dialect --canonicalize %s | FileCheck %s

// Test the folding of BrOp.

// CHECK-LABEL: func @br_folding(
p4hir.func @br_folding() -> !p4hir.int<32> {
  // CHECK-NEXT: %[[CST:.*]] = p4hir.const #int0_i32i
  // CHECK-NEXT: p4hir.return %[[CST]] : !i32i
  %c0_i32 = p4hir.const #p4hir.int<0> : !p4hir.int<32>
  p4hir.br ^bb1(%c0_i32 : !p4hir.int<32>)
^bb1(%x : !p4hir.int<32>):
  p4hir.return %x : !p4hir.int<32>
}

/// Test that pass-through successors of BrOp get folded.

// CHECK-LABEL: func @br_passthrough(
// CHECK-SAME: %[[ARG0:.*]]: !i32i, %[[ARG1:.*]]: !i32i
func.func @br_passthrough(%arg0 : !p4hir.int<32>, %arg1 : !p4hir.int<32>) -> (!p4hir.int<32>, !p4hir.int<32>) {
  "foo.switch"() [^bb1, ^bb2, ^bb3] : () -> ()

^bb1:
  // CHECK: ^bb1:
  // CHECK-NEXT: p4hir.br ^bb3(%[[ARG0]], %[[ARG1]] : !i32i, !i32i)

  p4hir.br ^bb2(%arg0 : !p4hir.int<32>)

^bb2(%arg2 : !p4hir.int<32>):
  p4hir.br ^bb3(%arg2, %arg1 : !p4hir.int<32>, !p4hir.int<32>)

^bb3(%arg4 : !p4hir.int<32>, %arg5 : !p4hir.int<32>):
  return %arg4, %arg5 : !p4hir.int<32>, !p4hir.int<32>
}
