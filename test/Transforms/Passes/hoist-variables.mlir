// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --p4hir-hoist-variables -split-input-file | FileCheck %s
// RUN: p4mlir-opt %s --p4hir-hoist-variables --p4hir-hoist-variables -split-input-file | FileCheck %s

// Variables declared in nested regions, at any depth, move to the start of the
// entry block of the function, in the order of their declarations. Their
// lifetimes start where they were declared and end with their scopes; their
// initializations stay in place. Variables already in the entry block stay in
// place, unmarked.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @f(
// CHECK-NEXT:    %[[A:.*]] = p4hir.variable ["a", init] : <!b8i>
// CHECK-NEXT:    %[[B:.*]] = p4hir.variable ["b"] : <!b8i>
// CHECK-NEXT:    %[[D:.*]] = p4hir.variable ["d"] : <!b8i>
// CHECK-NEXT:    %[[C:.*]] = p4hir.variable ["c"] : <!b8i>
// CHECK-NEXT:    %[[ONE:.*]] = p4hir.const
// CHECK-NEXT:    %[[E:.*]] = p4hir.variable ["e"] : <!b8i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[A]] : <!b8i>
// CHECK-NEXT:      p4hir.assign %[[ONE]], %[[A]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[A]] : <!b8i>
// CHECK-NEXT:    }
// CHECK-NEXT:    p4hir.if %{{.*}} {
// CHECK-NEXT:      p4hir.lifetime_start %[[B]] : <!b8i>
// CHECK-NEXT:      p4hir.assign %[[ONE]], %[[B]] : <!b8i>
// CHECK-NEXT:      p4hir.scope {
// CHECK-NEXT:        p4hir.lifetime_start %[[D]] : <!b8i>
// CHECK-NEXT:        p4hir.assign %[[ONE]], %[[D]] : <!b8i>
// CHECK-NEXT:        p4hir.lifetime_end %[[D]] : <!b8i>
// CHECK-NEXT:      }
// CHECK-NEXT:      p4hir.lifetime_end %[[B]] : <!b8i>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      p4hir.lifetime_start %[[C]] : <!b8i>
// CHECK-NEXT:      p4hir.assign %[[ONE]], %[[C]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[C]] : <!b8i>
// CHECK-NEXT:    }
// CHECK-NEXT:    p4hir.assign %[[ONE]], %[[E]] : <!b8i>
// CHECK-NEXT:    p4hir.return

module {
  p4hir.func @f(%cond : !p4hir.bool) {
    %one = p4hir.const #p4hir.int<1> : !b8i
    %e = p4hir.variable ["e"] : <!b8i>
    p4hir.scope {
      %a = p4hir.variable ["a", init] : <!b8i>
      p4hir.assign %one, %a : <!b8i>
    }
    p4hir.if %cond {
      %b = p4hir.variable ["b"] : <!b8i>
      p4hir.assign %one, %b : <!b8i>
      p4hir.scope {
        %d = p4hir.variable ["d"] : <!b8i>
        p4hir.assign %one, %d : <!b8i>
      }
    } else {
      %c = p4hir.variable ["c"] : <!b8i>
      p4hir.assign %one, %c : <!b8i>
    }
    p4hir.assign %one, %e : <!b8i>
    p4hir.return
  }
}

// -----

// A scope may already contain a CFG. Mark each dominated exit, without ending
// the lifetime at an internal branch.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @scope_cfg(
// CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!b8i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.cond_br
// CHECK:        ^bb1:
// CHECK-NEXT:      %{{.*}} = p4hir.read %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.yield
// CHECK:        ^bb2:
// CHECK-NEXT:      %{{.*}} = p4hir.read %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.yield
module {
  p4hir.func @scope_cfg(%cond : !p4hir.bool) {
    p4hir.scope {
      %x = p4hir.variable ["x"] : <!b8i>
      p4hir.cond_br %cond ^left, ^right
    ^left:
      %l = p4hir.read %x : <!b8i>
      p4hir.yield
    ^right:
      %r = p4hir.read %x : <!b8i>
      p4hir.yield
    }
    p4hir.return
  }
}

// -----

// When some paths bypass the declaration, no end marker is added at the join.
// The storage stays live conservatively, and the start still resets the value
// each time the declaration executes.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @conditional_declaration(
// CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!b8i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.cond_br
// CHECK:        ^bb1:
// CHECK-NEXT:      p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK-NEXT:      %{{.*}} = p4hir.read %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.br
// CHECK:        ^bb2:
// CHECK-NOT:       p4hir.lifetime_end
// CHECK-NEXT:      p4hir.yield
module {
  p4hir.func @conditional_declaration(%cond : !p4hir.bool) {
    p4hir.scope {
      p4hir.cond_br %cond ^declare, ^join
    ^declare:
      %x = p4hir.variable ["x"] : <!b8i>
      %v = p4hir.read %x : <!b8i>
      p4hir.br ^join
    ^join:
      p4hir.yield
    }
    p4hir.return
  }
}

// -----

// A variable declared in a loop body is allocated once for the whole loop. Its
// lifetime starts and ends on every iteration, and it is initialized on every
// iteration.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @loop(
// CHECK-NEXT:    %[[T:.*]] = p4hir.variable ["t", init] : <!b8i>
// CHECK-NEXT:    %[[ZERO:.*]] = p4hir.const
// CHECK-NEXT:    %[[I:.*]] = p4hir.variable ["i", init] : <!b8i>
// CHECK-NEXT:    p4hir.assign %[[ZERO]], %[[I]] : <!b8i>
// CHECK-NEXT:    p4hir.for : cond {
// CHECK:         } body {
// CHECK-NEXT:      p4hir.lifetime_start %[[T]] : <!b8i>
// CHECK-NEXT:      p4hir.assign %[[ZERO]], %[[T]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[T]] : <!b8i>
// CHECK-NEXT:      p4hir.yield
// CHECK-NEXT:    } updates {

module {
  p4hir.func @loop(%n : !b8i) {
    %zero = p4hir.const #p4hir.int<0> : !b8i
    %i = p4hir.variable ["i", init] : <!b8i>
    p4hir.assign %zero, %i : <!b8i>
    p4hir.for : cond {
      %vi = p4hir.read %i : <!b8i>
      %lt = p4hir.cmp(lt, %vi : !b8i, %n : !b8i)
      p4hir.condition %lt
    } body {
      %t = p4hir.variable ["t", init] : <!b8i>
      p4hir.assign %zero, %t : <!b8i>
      p4hir.yield
    } updates {
      %vi = p4hir.read %i : <!b8i>
      %inc = p4hir.binop(add, %vi, %n) : !b8i
      p4hir.assign %inc, %i : <!b8i>
      p4hir.yield
    }
    p4hir.return
  }
}

// -----

// A variable whose lifetime is already marked is hoisted without new markers.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.func @marked(
// CHECK-NEXT:    %[[X:.*]] = p4hir.variable ["x"] : <!b8i>
// CHECK-NEXT:    p4hir.scope {
// CHECK-NEXT:      p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK-NEXT:      p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NEXT:    }

module {
  p4hir.func @marked() {
    p4hir.scope {
      %x = p4hir.variable ["x"] : <!b8i>
      p4hir.lifetime_start %x : <!b8i>
      p4hir.lifetime_end %x : <!b8i>
    }
    p4hir.return
  }
}

// -----

// In controls and parsers, variables are hoisted to the entry block of their
// action, apply block or parser state, not to the body of the control or parser.
// Control-local variables are left in place.

!b8i = !p4hir.bit<8>

// CHECK-LABEL: p4hir.control @c()() {
// CHECK-NEXT:    p4hir.func action @act() {
// CHECK-NEXT:      %[[X:.*]] = p4hir.variable ["x"] : <!b8i>
// CHECK-NEXT:      %[[ONE:.*]] = p4hir.const
// CHECK-NEXT:      p4hir.scope {
// CHECK-NEXT:        p4hir.lifetime_start %[[X]] : <!b8i>
// CHECK-NEXT:        p4hir.assign %[[ONE]], %[[X]] : <!b8i>
// CHECK-NEXT:        p4hir.lifetime_end %[[X]] : <!b8i>
// CHECK-NEXT:      }
// CHECK-NEXT:      p4hir.return
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[LOCAL:.*]] = p4hir.variable ["local"] : <!b8i>
// CHECK-NEXT:    p4hir.control_local @local = %[[LOCAL]] : !p4hir.ref<!b8i>
// CHECK-NEXT:    p4hir.control_apply {
// CHECK-NEXT:      %[[Y:.*]] = p4hir.variable ["y"] : <!b8i>
// CHECK-NEXT:      %[[TWO:.*]] = p4hir.const
// CHECK-NEXT:      p4hir.scope {
// CHECK-NEXT:        p4hir.lifetime_start %[[Y]] : <!b8i>
// CHECK-NEXT:        p4hir.assign %[[TWO]], %[[Y]] : <!b8i>
// CHECK-NEXT:        p4hir.lifetime_end %[[Y]] : <!b8i>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: p4hir.parser @p()() {
// CHECK-NEXT:    p4hir.state @start {
// CHECK-NEXT:      %[[S:.*]] = p4hir.variable ["s"] : <!b8i>
// CHECK-NEXT:      %[[THREE:.*]] = p4hir.const
// CHECK-NEXT:      p4hir.scope {
// CHECK-NEXT:        p4hir.lifetime_start %[[S]] : <!b8i>
// CHECK-NEXT:        p4hir.assign %[[THREE]], %[[S]] : <!b8i>
// CHECK-NEXT:        p4hir.lifetime_end %[[S]] : <!b8i>
// CHECK-NEXT:      }
// CHECK-NEXT:      p4hir.transition to @accept

module {
  p4hir.control @c()() {
    p4hir.func action @act() {
      %one = p4hir.const #p4hir.int<1> : !b8i
      p4hir.scope {
        %x = p4hir.variable ["x"] : <!b8i>
        p4hir.assign %one, %x : <!b8i>
      }
      p4hir.return
    }
    %local = p4hir.variable ["local"] : <!b8i>
    p4hir.control_local @local = %local : !p4hir.ref<!b8i>
    p4hir.control_apply {
      %two = p4hir.const #p4hir.int<2> : !b8i
      p4hir.scope {
        %y = p4hir.variable ["y"] : <!b8i>
        p4hir.assign %two, %y : <!b8i>
      }
    }
  }

  p4hir.parser @p()() {
    p4hir.state @start {
      %three = p4hir.const #p4hir.int<3> : !b8i
      p4hir.scope {
        %s = p4hir.variable ["s"] : <!b8i>
        p4hir.assign %three, %s : <!b8i>
      }
      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }
}
