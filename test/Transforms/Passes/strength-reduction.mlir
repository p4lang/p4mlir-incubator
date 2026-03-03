// RUN: p4mlir-opt --p4hir-strength-reduction %s | FileCheck %s

!b32 = !p4hir.bit<32>

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_b32(!b32)

  // CHECK-LABEL: p4hir.func @mul2
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @mul2(%x: !b32) {
    // CHECK: %[[C1:.*]] = p4hir.const #int1_b32i
    // CHECK: %[[SHL:.*]] = p4hir.shl(%[[ARG]], %[[C1]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[SHL]])
    %c2 = p4hir.const #p4hir.int<2> : !b32
    %mul = p4hir.binop(mul, %x, %c2) : !b32
    p4hir.call @blackhole_b32(%mul) : (!b32) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @mul3
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @mul3(%x: !b32) {
    // CHECK: %[[C3:.*]] = p4hir.const #int3_b32i
    // CHECK: %[[MUL:.*]] = p4hir.binop(mul, %[[ARG]], %[[C3]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[MUL]])
    %c3 = p4hir.const #p4hir.int<3> : !b32
    %mul = p4hir.binop(mul, %x, %c3) : !b32
    p4hir.call @blackhole_b32(%mul) : (!b32) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @div2
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @div2(%x: !b32) {
    // CHECK: %[[C1:.*]] = p4hir.const #int1_b32i
    // CHECK: %[[SHR:.*]] = p4hir.shr(%[[ARG]], %[[C1]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[SHR]])
    %c2 = p4hir.const #p4hir.int<2> : !b32
    %div = p4hir.binop(div, %x, %c2) : !b32
    p4hir.call @blackhole_b32(%div) : (!b32) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @div3
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @div3(%x: !b32) {
    // CHECK: %[[C3:.*]] = p4hir.const #int3_b32i
    // CHECK: %[[DIV:.*]] = p4hir.binop(div, %[[ARG]], %[[C3]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[DIV]])
    %c3 = p4hir.const #p4hir.int<3> : !b32
    %div = p4hir.binop(div, %x, %c3) : !b32
    p4hir.call @blackhole_b32(%div) : (!b32) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @mod2
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @mod2(%x: !b32) {
    // CHECK: %[[C1:.*]] = p4hir.const #int1_b32i
    // CHECK: %[[AND:.*]] = p4hir.binop(and, %[[ARG]], %[[C1]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[AND]])
    %c2 = p4hir.const #p4hir.int<2> : !b32
    %mod = p4hir.binop(mod, %x, %c2) : !b32
    p4hir.call @blackhole_b32(%mod) : (!b32) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @mod3
  // CHECK-SAME: (%[[ARG:.*]]: !b32i)
  p4hir.func @mod3(%x: !b32) {
    // CHECK: %[[C3:.*]] = p4hir.const #int3_b32i
    // CHECK: %[[MOD:.*]] = p4hir.binop(mod, %[[ARG]], %[[C3]]{{.*}})
    // CHECK: p4hir.call @blackhole_b32 (%[[MOD]])
    %c3 = p4hir.const #p4hir.int<3> : !b32
    %mod = p4hir.binop(mod, %x, %c3) : !b32
    p4hir.call @blackhole_b32(%mod) : (!b32) -> ()
    p4hir.return
  }
}
