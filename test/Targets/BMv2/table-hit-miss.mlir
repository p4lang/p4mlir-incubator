// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s
!b8i = !p4hir.bit<8>
#int0_b8i = #p4hir.int<0> : !b8i
module {
  %c0_b8i = p4hir.const #int0_b8i
  bmv2ir.header_instance @hdr_out_arg_var_0 : !bmv2ir.header<"data_t", [c1:!p4hir.bit<8>, c2:!p4hir.bit<8>, c3:!p4hir.bit<8>, r1:!p4hir.bit<32>, r2:!p4hir.bit<32>, r3:!p4hir.bit<32>, b1:!p4hir.bit<8>, b2:!p4hir.bit<8>, b3:!p4hir.bit<8>, b4:!p4hir.bit<8>, b5:!p4hir.bit<8>, b6:!p4hir.bit<8>], max_length = 21>
  bmv2ir.header_instance @Headers_data : !bmv2ir.header<"data_t", [c1:!p4hir.bit<8>, c2:!p4hir.bit<8>, c3:!p4hir.bit<8>, r1:!p4hir.bit<32>, r2:!p4hir.bit<32>, r3:!p4hir.bit<32>, b1:!p4hir.bit<8>, b2:!p4hir.bit<8>, b3:!p4hir.bit<8>, b4:!p4hir.bit<8>, b5:!p4hir.bit<8>, b6:!p4hir.bit<8>], max_length = 21>
  bmv2ir.parser @parser init_state @parser::@start {
    bmv2ir.state @start
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @parser::@accept
    }
     parser_ops {
    }
    bmv2ir.state @accept
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default
    }
     parser_ops {
    }
  }
  bmv2ir.pipeline @ingress init_table @ingress::@cond_node_0 {
    p4hir.func action @NoAction_1() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @noop() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @noop_1() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @noop_2() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @setb1(%arg0: !b8i {p4hir.annotations = {name = "b1"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b1_1"}) annotations {name = "ingress.setb1"} {
      %0 = bmv2ir.field @Headers_data["b1"] -> !b8i
      bmv2ir.assign %arg0 : !b8i to %0 : !b8i
      p4hir.return
    }
    p4hir.func action @setb2(%arg0: !b8i {p4hir.annotations = {name = "b2"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b2_1"}) annotations {name = "ingress.setb2"} {
      %0 = bmv2ir.field @Headers_data["b2"] -> !b8i
      bmv2ir.assign %arg0 : !b8i to %0 : !b8i
      p4hir.return
    }
    p4hir.func action @setb3(%arg0: !b8i {p4hir.annotations = {name = "b3"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b3_1"}) annotations {name = "ingress.setb3"} {
      %0 = bmv2ir.field @Headers_data["b3"] -> !b8i
      bmv2ir.assign %arg0 : !b8i to %0 : !b8i
      p4hir.return
    }
    bmv2ir.table @t1_0
     actions [@ingress::@setb1, @ingress::@noop]
     next_tables #bmv2ir.hit_miss<hit @ingress::@t3_0 miss @ingress::@t2_0>
     type  simple
     match_type  exact
     keys [#bmv2ir.table_key<type exact, header @Headers_data["r1"]>]
     support_timeout false
     default_entry <action @ingress::@noop, action_const true, action_entry_const true>
     size 1024
    bmv2ir.table @t2_0
     actions [@ingress::@setb2, @ingress::@noop_1]
     next_tables [#bmv2ir.action_table<@ingress::@setb2 : @ingress::@t3_0>, #bmv2ir.action_table<@ingress::@noop_1 : @ingress::@t3_0>]
     type  simple
     match_type  exact
     keys [#bmv2ir.table_key<type exact, header @Headers_data["r2"]>]
     support_timeout false
     default_entry <action @ingress::@noop_1, action_const true, action_entry_const true>
     size 1024
    bmv2ir.table @t3_0
     actions [@ingress::@setb3, @ingress::@noop_2, @ingress::@NoAction_1]
     next_tables [#bmv2ir.action_table<@ingress::@setb3>, #bmv2ir.action_table<@ingress::@noop_2>, #bmv2ir.action_table<@ingress::@NoAction_1>]
     type  simple
     match_type  exact
     keys [#bmv2ir.table_key<type exact, header @Headers_data["b1"]>, #bmv2ir.table_key<type exact, header @Headers_data["b2"]>]
     support_timeout false
     default_entry <action @ingress::@NoAction_1, action_const true, action_entry_const true>
     size 1024
    bmv2ir.conditional @cond_node_0 then @ingress::@t1_0 expr {
      %0 = bmv2ir.field @Headers_data["c1"] -> !b8i
      %eq = p4hir.cmp(eq, %0 : !b8i, %c0_b8i : !b8i)
      bmv2ir.yield %eq : !p4hir.bool
    }
  }
  bmv2ir.pipeline @egress {
  }
  bmv2ir.deparser @deparser order []
}

// CHECK:      "conditionals": [
// CHECK:        {
// CHECK:          "expression": {
// CHECK:            "type": "expression",
// CHECK:            "value": {
// CHECK:              "left": {
// CHECK:                "type": "field",
// CHECK:                "value": [
// CHECK:                  "Headers_data",
// CHECK:                  "c1"
// CHECK:                ]
// CHECK:              },
// CHECK:              "op": "==",
// CHECK:              "right": {
// CHECK:                "type": "hexstr",
// CHECK:                "value": "0x00"
// CHECK:              }
// CHECK:            }
// CHECK:          },
// CHECK:          "false_next": null,
// CHECK:          "id": 0,
// CHECK:          "name": "cond_node_0",
// CHECK:          "true_next": "t1_0"
// CHECK:        }
// CHECK:      ],


// CHECK:        {
// CHECK:          "action_ids": [
// CHECK:            4,
// CHECK:            1
// CHECK:          ],
// CHECK:          "actions": [
// CHECK:            "ingress.setb1",
// CHECK:            "ingress.noop"
// CHECK:          ],
// CHECK:          "default_entry": {
// CHECK:            "action_const": true,
// CHECK:            "action_data": [],
// CHECK:            "action_entry_const": true,
// CHECK:            "action_id": 1
// CHECK:          },
// CHECK:          "id": 0,
// CHECK:          "key": [
// CHECK:            {
// CHECK:              "match_type": "exact",
// CHECK:              "target": [
// CHECK:                "Headers_data",
// CHECK:                "r1"
// CHECK:              ]
// CHECK:            }
// CHECK:          ],
// CHECK:          "match_type": "exact",
// CHECK:          "max_size": 1024,
// CHECK:          "name": "t1_0",
// CHECK:          "next_tables": {
// CHECK:            "__HIT__": "t3_0",
// CHECK:            "__MISS__": "t2_0"
// CHECK:          },
// CHECK:          "support_timeout": false,
// CHECK:          "type": "simple"
// CHECK:        },
