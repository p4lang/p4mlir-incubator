// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | diff -u - %s.ref
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | p4test -

// CHECK: #p_ctorval = #p4hir.ctor_param<@p, "ctorval"> : !p4hir.bool
// CHECK:  p4hir.parser @p(%arg0: !i10i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "sinit"})(ctorval: !p4hir.bool)
// CHECK   %{{.*}} = p4hir.const ["ctorval"] #p_ctorval

parser p(in int<10> sinit)(bool ctorval) {
    int<10> s = ctorval ? 0 : sinit;

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

// The latter is not supported yet (and likely will not be supported)
/*
parser p2(in int<10> sinit)(bool ctorval) {
    const int<10> c = ctorval ? 10s10: 10s42;
    int<10> s = ctorval ? 0 : sinit;

    state start {
        s = 1 + (ctorval ? 123 : c);
        transition next;
    }
    
    state next {   
        s = 2;
        transition accept;
    }

    state drop {}
}*/
