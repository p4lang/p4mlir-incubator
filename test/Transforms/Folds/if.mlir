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

}
