// RUN: p4mlir-opt --p4hir-switch-elimination %s | FileCheck %s

!b32i = !p4hir.bit<32>
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int16_b32i = #p4hir.int<16> : !b32i
#int32_b32i = #p4hir.int<32> : !b32i

// Test: Switch on arbitrary expression is transformed to table-based switch on action_run.
// CHECK-LABEL: p4hir.control @c
module {
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "b"})() {
    // Verify action functions are created with hidden annotation.
    // CHECK-DAG: p4hir.func action @_switch_{{[0-9]+}}_case_0() annotations {hidden} {
    // CHECK-DAG:   p4hir.return
    // CHECK-DAG: }
    // CHECK-DAG: p4hir.func action @_switch_{{[0-9]+}}_case_1() annotations {hidden} {
    // CHECK-DAG:   p4hir.return
    // CHECK-DAG: }
    // CHECK-DAG: p4hir.func action @_switch_{{[0-9]+}}_default() annotations {hidden} {
    // CHECK-DAG:   p4hir.return
    // CHECK-DAG: }

    // Verify table structure with key, actions, default, and entries.
    // CHECK: p4hir.table @_switch_{{[0-9]+}}_table annotations {hidden} {
    // CHECK:   p4hir.table_key
    // CHECK:     p4hir.match_key #exact
    // CHECK:   p4hir.table_actions {
    // CHECK:     p4hir.table_action @_switch_{{[0-9]+}}_case_0() {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_case_0 () : () -> ()
    // CHECK:     }
    // CHECK:     p4hir.table_action @_switch_{{[0-9]+}}_case_1() {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_case_1 () : () -> ()
    // CHECK:     }
    // CHECK:     p4hir.table_action @_switch_{{[0-9]+}}_default() {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_default () : () -> ()
    // CHECK:     }
    // CHECK:   }
    // CHECK:   p4hir.table_default_action {
    // CHECK:     p4hir.call @c::@_switch_{{[0-9]+}}_default () : () -> ()
    // CHECK:   }
    // CHECK:   p4hir.table_entries const {
    // CHECK:     p4hir.table_entry #p4hir.aggregate<[#int16_b32i]> : tuple<!b32i> {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_case_0 () : () -> ()
    // CHECK:     }
    // CHECK:     p4hir.table_entry #p4hir.aggregate<[#int32_b32i]> : tuple<!b32i> {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_case_0 () : () -> ()
    // CHECK:     }
    // CHECK:     p4hir.table_entry #p4hir.aggregate<[#int2_b32i]> : tuple<!b32i> {
    // CHECK:       p4hir.call @c::@_switch_{{[0-9]+}}_case_1 () : () -> ()
    // CHECK:     }
    // CHECK:   }
    // CHECK: }

    // Verify table is applied directly with the condition (no intermediate variable).
    // CHECK: p4hir.control_apply {
    // CHECK:   %{{.*}} = p4hir.read %arg0 : <!b32i>
    // CHECK-NEXT: %{{.*}} = p4hir.table_apply @c::@_switch_{{[0-9]+}}_table with key{{.*}}%{{.*}}
    // CHECK:   %{{.*}} = p4hir.struct_extract %{{.*}}["action_run"]
    // CHECK:   p4hir.switch (%{{.*}}{{.*}}) {
    // CHECK:     p4hir.case(equal, [#{{[_a-z0-9]+}}_switch_{{[0-9]+}}_case_0]) {
    // CHECK:       %{{.*}} = p4hir.const #int1_b32i
    // CHECK:       p4hir.assign %{{.*}}, %arg0 : <!b32i>
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:     p4hir.case(equal, [#{{[_a-z0-9]+}}_switch_{{[0-9]+}}_case_1]) {
    // CHECK:       %{{.*}} = p4hir.const #int2_b32i
    // CHECK:       p4hir.assign %{{.*}}, %arg0 : <!b32i>
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:     p4hir.case(default, [#{{[_a-z0-9]+}}_switch_{{[0-9]+}}_default]) {
    // CHECK:       %{{.*}} = p4hir.const #int3_b32i
    // CHECK:       p4hir.assign %{{.*}}, %arg0 : <!b32i>
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:     p4hir.yield
    // CHECK:   }
    // CHECK: }
    
    p4hir.control_apply {
      %val = p4hir.read %arg0 : <!b32i>
      p4hir.switch (%val : !b32i) {
        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
          %c1_b32i = p4hir.const #int1_b32i
          p4hir.assign %c1_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(equal, [#int2_b32i]) {
          %c2_b32i = p4hir.const #int2_b32i
          p4hir.assign %c2_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(default, []) {
          %c3_b32i = p4hir.const #int3_b32i
          p4hir.assign %c3_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.yield
      }
    }
  }
}

// -----

// Test: Switches which are already on action_run (from `switch (t.apply().action_run)`)
// are NOT transformed again by the switch-elimination pass.
//
// CHECK-LABEL: p4hir.control @c_action_run
!anon = !p4hir.enum<a1, a2>
!t = !p4hir.struct<"t", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#anon_a1 = #p4hir.enum_field<a1, !anon> : !anon

module {
  p4hir.control @c_action_run()() {
    // The table should remain unchanged - no new tables created.
    // CHECK: p4hir.table @t {
    // CHECK:   p4hir.table_actions {
    // CHECK:     p4hir.table_action @a1() {
    // CHECK:       p4hir.call @c_action_run::@a1 () : () -> ()
    // CHECK:     }
    // CHECK:     p4hir.table_action @a2() {
    // CHECK:       p4hir.call @c_action_run::@a2 () : () -> ()
    // CHECK:     }
    // CHECK:   }
    // CHECK:   p4hir.table_default_action {
    // CHECK:     p4hir.call @c_action_run::@a1 () : () -> ()
    // CHECK:   }
    // CHECK: }
    // CHECK-NOT: p4hir.table @_switch
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @a1() {
          p4hir.call @c_action_run::@a1 () : () -> ()
        }
        p4hir.table_action @a2() {
          p4hir.call @c_action_run::@a2 () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @c_action_run::@a1 () : () -> ()
      }
    }

    p4hir.func action @a1() {
      p4hir.return
    }
    p4hir.func action @a2() {
      p4hir.return
    }

    // The switch on action_run should remain unchanged.
    // CHECK: p4hir.control_apply {
    // CHECK:   %{{.*}} = p4hir.table_apply @c_action_run::@t with key()
    // CHECK:   %{{.*}} = p4hir.struct_extract %{{.*}}["action_run"]
    // CHECK:   p4hir.switch (%{{.*}} : !anon) {
    // CHECK:     p4hir.case(equal, [#anon_a1]) {
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:     p4hir.case(default, []) {
    // CHECK:       p4hir.yield
    // CHECK:     }
    // CHECK:     p4hir.yield
    // CHECK:   }
    // CHECK: }
    p4hir.control_apply {
      %t_apply_result = p4hir.table_apply @c_action_run::@t with key()
        : () -> !t
      %action_run = p4hir.struct_extract %t_apply_result["action_run"] : !t
      p4hir.switch (%action_run : !anon) {
        p4hir.case(equal, [#anon_a1]) {
          p4hir.yield
        }
        p4hir.case(default, []) {
          p4hir.yield
        }
        p4hir.yield
      }
    }
  }
}
