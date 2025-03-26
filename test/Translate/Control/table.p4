// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

match_kind {
    exact,
    lpm
}

header MyHeader {
    int<16> f1;
}

extern Ext {
    Ext(int<16> tmp);
}

extern int<16> baz(in int<16> arg);

action bak() {}

// CHECK-LABEL:   p4hir.control @Pipe
control Pipe(in MyHeader arg1, in int<16> arg2, inout int<16> arg3)(int<16> ctr_arg1) {
    action foo(in int<16> x1, inout int<16> x2) {}
    const int<16> cst = 3;
    MyHeader local_hdr = {23};

    action bar(int<16> x1, int<16> x2, int<16> x3) {
        local_hdr.f1 = arg2;
    }

    int<16> test = 52;

    table myTable {
        key = {
            arg1.f1 - 1 : exact @name("arg1.key");
            arg3 & 0xAABB : lpm @name("arg3.key");
            cst : lpm @name("cst.key");
            baz(test) : exact @name("baz.key");
            local_hdr.f1 : lpm @name("local_hdr.key");
        }
        // CHECK: p4hir.table_action @bar(%[[arg3:.*]]: !i16i {p4hir.param_name = "x1"}, %[[arg4:.*]]: !i16i {p4hir.param_name = "x2"}, %[[arg5:.*]]: !i16i {p4hir.param_name = "x3"}) {
        // CHECK:   p4hir.call @bar (%[[arg3]], %[[arg4]], %[[arg5]]) : (!i16i, !i16i, !i16i) -> ()
        // CHECK: }        
        actions = { foo(arg2, arg3); bar; bak(); }
        const default_action = bak();
        prop1 = 42;
        prop2 = baz(cst);
        prop3 = baz(ctr_arg1 + 3);
        prop5 = baz(test);
    }

    apply {
        myTable.apply();
        test = test + 1;
        local_hdr.f1 = 3;
    }
}
