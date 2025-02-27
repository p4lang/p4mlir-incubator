// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: #p_ctorval = #p4hir.ctor_param<@p, "ctorval"> : !p4hir.bool
// CHECK:  p4hir.parser @p(%arg0: !i10i)(ctorval: !p4hir.bool) {
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
