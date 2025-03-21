// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL: p4hir.control @c(
// CHECK-SAME: %[[arg0:.*]]: !p4hir.ref<!b32i>
control c(inout bit<32> b) {
    apply {
// CHECK:   p4hir.control_apply {
// CHECK:      %[[val:.*]] = p4hir.read %[[arg0]] : <!b32i>
// CHECK:      p4hir.switch (%[[val]] : !b32i) {
// CHECK:        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
//                 ...
// CHECK:          p4hir.yield
// CHECK:        }
// CHECK:        p4hir.case(equal, [#int64_b32i]) {
//                 ...                 
// CHECK:        }
// CHECK:        p4hir.case(default, [#int92_b32i]) {
//                 ...
// CHECK:          p4hir.yield
// CHECK:        }
// CHECK:        p4hir.yield
// CHECK:     }
        switch (b) {
            16:
            32: { b = 1; }
            64: { b = 2; }
            92:
            default: { b = 3; }
        }
    }
}

control ct(inout bit<32> b);
package top(ct _c);
top(c()) main;
