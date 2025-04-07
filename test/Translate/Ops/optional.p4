// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

control c(out bool b) {
  apply {
    b = false;
  }
}

parser par(out bool b) {
    state start {
        b = false;
        transition accept;
    }
}

control ce(out bool b);
parser pe(out bool b);
package top(pe _p, ce _e, @optional ce _e1);

top(_e = c(),
    _p = par()) main;

// CHECK: !c = !p4hir.control<"c", (!p4hir.ref<!p4hir.bool>)>
// CHECK: !ce = !p4hir.control<"ce", (!p4hir.ref<!p4hir.bool>)>
// CHECK: !par = !p4hir.parser<"par", (!p4hir.ref<!p4hir.bool>)>
// CHECK: !pe = !p4hir.parser<"pe", (!p4hir.ref<!p4hir.bool>)>
// CHECK: !top = !p4hir.package<"top">
// CHECK:  %[[uninitialized:.*]] = p4hir.uninitialized : !ce
// CHECK:  p4hir.instantiate @top(%{{.*}}, %{{.}}, %[[uninitialized]]) as "main" : (!par, !c, !ce) -> !top
