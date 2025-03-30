// RUN: p4mlir-opt %s --verify-diagnostics

!b32i = !p4hir.bit<32>
#int0_b32i = #p4hir.int<0> : !b32i

// expected-error@+6 {{'p4hir.for' op expected condition region to terminate with 'p4hir.condition'}}
module {
  %0 = p4hir.const #int0_b32i
  p4hir.scope {
    %i = p4hir.variable ["i", init] : <!b32i>
    p4hir.assign %0, %i : <!b32i>
    p4hir.for : cond {
      p4hir.yield
    } body {
      p4hir.yield
    } updates {
      p4hir.yield
    }
  }
}
