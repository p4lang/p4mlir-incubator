// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s


struct empty {}

parser p(in empty e, in int<10> sinit) {
    int<10> s = sinit;

    state start {
        s = 1;
        transition next;
    }
    
    state next {   
        s = 2;
        transition accept;
    }

    state drop {}
}

// CHECK-LABEL: p4hir.parser @p(%arg0: !empty {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "e"}, %arg1: !i10i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "sinit"})()
// CHECK:    p4hir.state @start {
// CHECK:      p4hir.transition to @p::@next
// CHECK:    p4hir.state @next {
// CHECK:      p4hir.transition to @p::@accept
// CHECK:    p4hir.state @drop {
// CHECK-NEXT:      p4hir.transition to @p::@reject
// CHECK:    p4hir.state @accept {
// CHECK-NEXT:      p4hir.parser_accept
// CHECK:    p4hir.state @reject {
// CHECK-NEXT:      p4hir.parser_reject
// CHECK:    p4hir.transition to @p::@start
