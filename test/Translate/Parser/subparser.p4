// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

struct empty {}

parser subparser(in empty e) {
    state start {
        transition accept;
    }
}

parser subparser2(in empty e)(bool ctorArg) {
    state start {
        transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @p
// CHECK: p4hir.instantiate @subparser() as "sp" : () -> !p4hir.parser<"subparser", (!empty)>
// CHECK: %[[false:.*]] = p4hir.const #false
// CHECK: p4hir.instantiate @subparser2(%[[false]]) as "sp2" : (!p4hir.bool) -> !p4hir.parser<"subparser2", (!empty)>

parser p(in empty e, in int<10> sinit) {
    int<10> s = sinit;
    subparser() sp;
    subparser2(false) sp2;

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
