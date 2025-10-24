// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | diff -u - %s.ref
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | p4test -

// CHECK: p4hir.func @log(!string {p4hir.dir = #undir, p4hir.param_name = "s"})
extern void log(string s);

// CHECK-LABEL: @test
action test() {
    // CHECK: %[[cst:.*]] = p4hir.const "This is a message" : !string
    log("This is a message");
    // CHECK: p4hir.call @log (%[[cst]]) : (!string) -> ()
}
