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

parser subparser3(inout int<10> s, out bool matched)() {
    state start {
        transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @p(%arg0: !empty, %arg1: !i10i)()
parser p(in empty e, in int<10> sinit) {
    int<10> s = sinit;
    subparser() sp;
    subparser2(false) sp2;
    subparser3() sp3;
// CHECK: %[[sp:.*]] = p4hir.instantiate @subparser() as "sp" : () -> !subparser
// CHECK: %[[false:.*]] = p4hir.const #false
// CHECK: %[[sp2:.*]] = p4hir.instantiate @subparser2(%[[false]]) as "sp2" : (!p4hir.bool) -> !subparser2
// CHECK: %[[sp3:.*]] = p4hir.instantiate @subparser3() as "sp3" : () -> !subparser3

    state start {
        s = 1;
        sp.apply(e);
// CHECK: p4hir.apply %[[sp]](%arg0) : !subparser
        transition next;
    }

    state next {
        s = 2;
        sp2.apply(e);
// CHECK: p4hir.apply %[[sp2]](%arg0) : !subparser2
// CHECK: p4hir.scope
// CHECK: p4hir.apply %[[sp3]](%s_inout_arg, %matched_out_arg) : !subparser3
        bool matched = false;
        sp3.apply(s, matched);

        transition accept;
    }

    state drop {}
}
