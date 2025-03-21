// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

match_kind {
  exact,
  ternary,
  lpm
}

// CHECK: !Annotated = !p4hir.extern<"Annotated" annotations {size = ["100"]}>

// CHECK:   p4hir.const ["b"] #int-1_b1i annotations {hidden}
@hidden const bit b = 1;

struct Foo {
  @match(ternary) bit<32> bar;
}

// CHECK-LABEL  p4hir.extern @Annotated annotations {size = ["100"]} {
// CHECK:    p4hir.func @Annotated() annotations {hidden, name = "annotated", pkginfo = {bar = "42", foo = 10 : i64}}
// CHECK:    p4hir.func @execute(!b8i {p4hir.annotations = {optional}, p4hir.dir = #undir, p4hir.param_name = "index"}) annotations {name = "exe"}

@size(100)
extern Annotated {
    @name("annotated") @hidden @pkginfo(foo=10, bar="42")
    Annotated();
    @name("exe")
    void execute(@optional bit<8> index);
}

extern Virtual {
    Virtual();
    void run(in bit<16> ix);  // internally calls f
    // CHECK: p4hir.func @f(!b16i {p4hir.dir = #in, p4hir.param_name = "ix"}) -> !b16i annotations {synchronous = @run}
    @synchronous(run) abstract bit<16> f(in bit<16> ix);
}

// CHECK:   p4hir.func @log(!b32i {p4hir.dir = #in, p4hir.param_name = "data"}) -> !b32i annotations {pure}
@pure
extern bit<32> log(in bit<32> data);

struct empty {}

// CHECK-LABEL:   p4hir.parser @p(%arg0: !empty {p4hir.dir = #in, p4hir.param_name = "e"}, %arg1: !i10i {p4hir.dir = #in, p4hir.param_name = "sinit"})() annotations {pkginfo = {bar = "42", foo = 10 : i64}}
@pkginfo(foo=10, bar="42")
parser p(in empty e, in int<10> sinit) {
// CHECK:     p4hir.variable ["s", init] annotations {name = "var.s"} : <!i10i>
    @name("var.s") int<10> s = sinit;

// CHECK:     p4hir.state @start annotations {name = "state.start"}
    @name("state.start")
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

// CHECK-LABEL:   p4hir.control @c(%arg0: !Foo {p4hir.dir = #in, p4hir.param_name = "ff"}, %arg1: !p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "bb"})() annotations {pkginfo = {bar = "42", foo = 10 : i64}}
@pkginfo(foo=10, bar="42")
control c(in Foo ff, bool bb) {
// CHECK:     p4hir.func action @a() annotations {hidden}
    @hidden action a() {}
    action b() {}
    action cc() {}

// CHECK:     p4hir.table @t annotations {name = "table.t"}
    @name("table.t") table t {
// CHECK:         p4hir.match_key #exact %c42_b10i : !b10i annotations {name = "exact.key"}    
        key = { 10w42 : exact @name("exact.key"); false : lpm; }
        actions = {
// CHECK: p4hir.table_action @a() {        
           a;
// CHECK: p4hir.table_action @b() annotations {tableonly} {           
           @tableonly b;
// CHECK: p4hir.table_action @cc() annotations {defaultonly} {           
           @defaultonly cc;
        }
// CHECK:      p4hir.table_default_action annotations {name = "bar"} {        
        @name("bar") default_action = cc;
// CHECK:      p4hir.table_entry "implementation" annotations {name = "foo"} {
        @name("foo") implementation = Annotated();
// CHECK:      p4hir.table_size #int42_infint annotations {name = "dummy.size"}        
        @name("dummy.size") size = 42;
    }

    apply {
// CHECK:      p4hir.scope annotations {unlikely} {    
        @unlikely {
        }

// CHECK:       p4hir.if %{{.*}} annotations {likely} {
        if (bb) @likely {
        }

// CHECK:       p4hir.if %{{.*}} annotations {likely} {
        if (bb) @likely {
// CHECK:      } else annotations {unlikely} {        
        } else @unlikely {
        }
    }
}
