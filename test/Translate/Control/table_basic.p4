// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

match_kind {
    exact,
    ternary,
    lpm
}

// CHECK-LABEL:  p4hir.extern @ActionProfile
// CHECK:     p4hir.func @ActionProfile(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"})
extern ActionProfile {
   ActionProfile(bit<32> size);
}

// CHECK-LABEL:  p4hir.control @c(%arg0: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg"})()
control c(in bit<32> arg) {
    // CHECK: p4hir.func action @a(%arg1: !b32i {p4hir.dir = #undir, p4hir.param_name = "carg"})
    action a(bit<32> carg) {}
    // CHECK: p4hir.func action @aa(%arg1: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg"}, %arg2: !i8i {p4hir.dir = #undir, p4hir.param_name = "carg"})
    action aa(in bit<32> arg, int<8> carg) {}
    // CHECK: p4hir.func action @b() {
    action b() {}

    // CHECK-LABEL: p4hir.table @t1 {
    table t1 {
    // CHECK: p4hir.table_actions {
    // CHECK:    p4hir.table_action @a(%arg1: !b32i {p4hir.param_name = "carg"}) {
    // CHECK:      p4hir.call @c::@a (%arg1) : (!b32i) -> ()
    // CHECK:    }
    // CHECK:    p4hir.table_action @b() {
    // CHECK:      p4hir.call @c::@b () : () -> ()
    // CHECK:    }
    // CHECK:  }
    // CHECK:  p4hir.table_default_action {
    // CHECK:    p4hir.call @c::@b () : () -> ()
    // CHECK:  }
        actions = { a; b; }
        default_action = b;
    }

    // CHECK-LABEL: p4hir.table @t2 {
    table t2 {
    // CHECK:  p4hir.table_key {
    // CHECK:    p4hir.match_key #exact %arg0 : !b32i annotations {name = "arg.key"}
    // CHECK:    %false = p4hir.const #false
    // CHECK:    p4hir.match_key #lpm %false : !p4hir.bool annotations {name = "false.key"}
    // CHECK:  }
        key = { arg : exact @name("arg.key"); false : lpm @name("false.key"); }
        actions = { a; b;  aa(arg); }
        default_action = b;
        size = 42;
     // CHECK:  %size = p4hir.table_size #int42_infint        
        largest_priority_wins = false;
      // CHECK: %largest_priority_wins = p4hir.table_entry "largest_priority_wins" {        
        priority_delta = 10;
      // CHECK: %priority_delta = p4hir.table_entry "priority_delta" {        
        some_entry = 10;
        implementation = ActionProfile(32);
       // CHECK:%implementation = p4hir.table_entry "implementation" {
       // CHECK: %ActionProfile = p4hir.construct @ActionProfile (%{{.*}} : !b32i) : !ActionProfile
       // CHECK: p4hir.yield %ActionProfile : !ActionProfile
       // CHECK:} : !ActionProfile        
    }

    apply {
        if (t1.apply().hit) {
            t2.apply();
        }
    }
    // CHECK: p4hir.control_apply {
    // CHECK:   %[[t1_apply_result:.*]] = p4hir.table_apply @c::@t1 : !t1
    // CHECK:   %[[hit:.*]] = p4hir.struct_extract %[[t1_apply_result]]["hit"] : !t1
    // CHECK:   p4hir.if %[[hit]] {
    // CHECK:     %{{.*}}= p4hir.table_apply @c::@t2 : !t2
    // CHECK:   }
}
