// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: p4hir.func @log(!string {p4hir.dir = #undir, p4hir.param_name = "s"})
extern void log(string s);

// CHECK-LABEL: @test
action test() {
    // CHECK: %[[cst:.*]] = p4hir.const "This is a message" : !string
    log("This is a message");
    // CHECK: p4hir.call @log (%[[cst]]) : (!string) -> ()
}
