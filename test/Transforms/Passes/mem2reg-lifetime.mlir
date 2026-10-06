// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(any(mem2reg))' %s | FileCheck %s

// A variable comes into existence holding the default value of its type, so
// promotion turns the start of its lifetime into a definition of that value.
// Lifetime markers do not block promotion and are removed.

!b8i = !p4hir.bit<8>

module {
  // The value assigned in a previous lifetime is gone.
  // CHECK-LABEL: p4hir.func @restart
  // CHECK-NOT:     p4hir.variable
  // CHECK-NOT:     p4hir.lifetime
  // CHECK:         p4hir.const #int5_b8i
  // CHECK-NEXT:    %[[ZERO:.*]] = p4hir.const #int0_b8i
  // CHECK-NEXT:    p4hir.return %[[ZERO]] : !b8i
  p4hir.func @restart() -> !b8i {
    %x = p4hir.variable ["x"] : <!b8i>
    p4hir.lifetime_start %x : <!b8i>
    %c5 = p4hir.const #p4hir.int<5> : !b8i
    p4hir.assign %c5, %x : <!b8i>
    p4hir.lifetime_end %x : <!b8i>
    p4hir.lifetime_start %x : <!b8i>
    %v = p4hir.read %x : <!b8i>
    p4hir.return %v : !b8i
  }

  // A variable whose lifetime starts in a loop body, e.g. one hoisted out of
  // it, holds the default value on every iteration: no value is carried over
  // from the previous one.
  // CHECK-LABEL: p4hir.func @loop
  // CHECK:         p4hir.br ^[[HEAD:bb[0-9]+]](%{{.*}} : !b8i)
  // CHECK-NEXT:  ^[[HEAD]](%[[ACC:.*]]: !b8i):
  // CHECK:         p4hir.cond_br %{{.*}} ^[[BODY:bb[0-9]+]], ^{{.*}}
  // CHECK-NEXT:  ^[[BODY]]:
  // CHECK-NEXT:    %[[ZERO:.*]] = p4hir.const #int0_b8i
  // CHECK-NEXT:    %[[ONE:.*]] = p4hir.const #int1_b8i
  // CHECK-NEXT:    %[[INC:.*]] = p4hir.binop(add, %[[ZERO]], %[[ONE]]) : !b8i
  // CHECK-NEXT:    %[[SUM:.*]] = p4hir.binop(add, %[[ACC]], %[[INC]]) : !b8i
  // CHECK-NEXT:    p4hir.br ^[[HEAD]](%[[SUM]] : !b8i)
  p4hir.func @loop(%n : !b8i) -> !b8i {
    %zero = p4hir.const #p4hir.int<0> : !b8i
    %acc = p4hir.variable ["acc", init] : <!b8i>
    p4hir.assign %zero, %acc : <!b8i>
    %t = p4hir.variable ["t"] : <!b8i>
    p4hir.br ^head
  ^head:
    %a = p4hir.read %acc : <!b8i>
    %lt = p4hir.cmp(lt, %a : !b8i, %n : !b8i)
    p4hir.cond_br %lt ^body, ^exit
  ^body:
    p4hir.lifetime_start %t : <!b8i>
    %tv = p4hir.read %t : <!b8i>
    %one = p4hir.const #p4hir.int<1> : !b8i
    %inc = p4hir.binop(add, %tv, %one) : !b8i
    p4hir.assign %inc, %t : <!b8i>
    %a2 = p4hir.read %acc : <!b8i>
    %sum = p4hir.binop(add, %a2, %inc) : !b8i
    p4hir.assign %sum, %acc : <!b8i>
    p4hir.lifetime_end %t : <!b8i>
    p4hir.br ^head
  ^exit:
    %r = p4hir.read %acc : <!b8i>
    p4hir.return %r : !b8i
  }
}
