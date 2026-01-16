// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s
!b12i = !p4hir.bit<12>
!b13i = !p4hir.bit<13>
!b16i = !p4hir.bit<16>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!packet_out = !p4hir.extern<"packet_out">
!validity_bit = !p4hir.validity.bit
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
#int-1_b8i = #p4hir.int<255> : !b8i
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
module {
  bmv2ir.header_instance @ingress2 : !bmv2ir.header<"standard_metadata_t", [ingress_port:!p4hir.bit<9>, egress_spec:!p4hir.bit<9>, egress_port:!p4hir.bit<9>, instance_type:!p4hir.bit<32>, packet_length:!p4hir.bit<32>, enq_timestamp:!p4hir.bit<32>, enq_qdepth:!p4hir.bit<19>, deq_timedelta:!p4hir.bit<32>, deq_qdepth:!p4hir.bit<19>, ingress_global_timestamp:!p4hir.bit<48>, egress_global_timestamp:!p4hir.bit<48>, mcast_grp:!p4hir.bit<16>, egress_rid:!p4hir.bit<16>, checksum_error:!p4hir.bit<1>, priority:!p4hir.bit<3>, _padding:!p4hir.bit<3>], max_length = 41>
  bmv2ir.header_instance @ingress1 : !bmv2ir.header<"ingress_metadata_t", [vrf:!p4hir.bit<12>, bd:!p4hir.bit<16>, nexthop_index:!p4hir.bit<16>, _padding:!p4hir.bit<4>], max_length = 6>
  bmv2ir.header_instance @ingress0_ipv4 : !bmv2ir.header<"ipv4_t", [version:!p4hir.bit<4>, ihl:!p4hir.bit<4>, diffserv:!p4hir.bit<8>, totalLen:!p4hir.bit<16>, identification:!p4hir.bit<16>, flags:!p4hir.bit<3>, fragOffset:!p4hir.bit<13>, ttl:!p4hir.bit<8>, protocol:!p4hir.bit<8>, hdrChecksum:!p4hir.bit<16>, srcAddr:!p4hir.bit<32>, dstAddr:!p4hir.bit<32>], max_length = 20>
  bmv2ir.pipeline @ingress init_table @ingress::@conditional_name0 {
    p4hir.func action @NoAction_3() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @NoAction_4() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @NoAction_5() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @NoAction_6() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @NoAction_7() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @set_vrf(%arg0: !b12i {p4hir.annotations = {name = "vrf"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "vrf_1"}) annotations {name = "ingress.set_vrf"} {
      %0 = bmv2ir.field @ingress1["vrf"] -> !b12i
      bmv2ir.assign %arg0 : !b12i to %0 : !b12i
      p4hir.return
    }
    p4hir.func action @on_miss_2() annotations {name = "ingress.on_miss"} {
      p4hir.return
    }
    p4hir.func action @on_miss_3() annotations {name = "ingress.on_miss"} {
      p4hir.return
    }
    p4hir.func action @on_miss_4() annotations {name = "ingress.on_miss"} {
      p4hir.return
    }
    p4hir.func action @fib_hit_nexthop(%arg0: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "nexthop_index_1"}) annotations {name = "ingress.fib_hit_nexthop"} {
      %c-1_b8i = p4hir.const #int-1_b8i
      %0 = bmv2ir.field @ingress1["nexthop_index"] -> !b16i
      bmv2ir.assign %arg0 : !b16i to %0 : !b16i
      %1 = bmv2ir.field @ingress0_ipv4["ttl"] -> !b8i
      %2 = bmv2ir.field @ingress0_ipv4["ttl"] -> !b8i
      %add = p4hir.binop(add, %2, %c-1_b8i) : !b8i
      bmv2ir.assign %add : !b8i to %1 : !b8i
      p4hir.return
    }
    p4hir.func action @fib_hit_nexthop_1(%arg0: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "nexthop_index_2"}) annotations {name = "ingress.fib_hit_nexthop"} {
      %c-1_b8i = p4hir.const #int-1_b8i
      %0 = bmv2ir.field @ingress1["nexthop_index"] -> !b16i
      bmv2ir.assign %arg0 : !b16i to %0 : !b16i
      %1 = bmv2ir.field @ingress0_ipv4["ttl"] -> !b8i
      %2 = bmv2ir.field @ingress0_ipv4["ttl"] -> !b8i
      %add = p4hir.binop(add, %2, %c-1_b8i) : !b8i
      bmv2ir.assign %add : !b8i to %1 : !b8i
      p4hir.return
    }
    p4hir.func action @set_egress_details(%arg0: !b9i {p4hir.annotations = {name = "egress_spec"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "egress_spec_1"}) annotations {name = "ingress.set_egress_details"} {
      %0 = bmv2ir.field @ingress2["egress_spec"] -> !b9i
      bmv2ir.assign %arg0 : !b9i to %0 : !b9i
      p4hir.return
    }
    p4hir.func action @set_bd(%arg0: !b16i {p4hir.annotations = {name = "bd"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "bd_0"}) annotations {name = "ingress.set_bd"} {
      %0 = bmv2ir.field @ingress1["bd"] -> !b16i
      bmv2ir.assign %arg0 : !b16i to %0 : !b16i
      p4hir.return
    }
    bmv2ir.table @bd_1
     actions [@ingress::@set_vrf, @ingress::@NoAction_3]
     next_tables [#bmv2ir.action_table<@ingress::@set_vrf : @ingress::@ipv4_fib_0>, #bmv2ir.action_table<@ingress::@NoAction_3 : @ingress::@ipv4_fib_0>]
     type simple
     match_type exact
     keys [#bmv2ir.table_key<type exact, header @ingress1["bd"] name = "foo.bar">]
     support_timeout false
     default_entry <action @ingress::@NoAction_3,  action_const true, action_entry_const true>
     size 65536
    bmv2ir.table @ipv4_fib_0
     actions [@ingress::@on_miss_2, @ingress::@fib_hit_nexthop, @ingress::@NoAction_4]
     next_tables [#bmv2ir.action_table<@ingress::@on_miss_2 : @ingress::@ipv4_fib_lpm_0>, #bmv2ir.action_table<@ingress::@fib_hit_nexthop : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@NoAction_4 : @ingress::@nexthop_0>]
     type simple
     match_type exact
     keys [#bmv2ir.table_key<type exact, header @ingress1["vrf"]>, #bmv2ir.table_key<type exact, header @ingress0_ipv4["dstAddr"]>]
     support_timeout false
     default_entry <action @ingress::@NoAction_4, action_const true, action_entry_const true>
     size 131072
    bmv2ir.table @ipv4_fib_lpm_0
     actions [@ingress::@on_miss_3, @ingress::@fib_hit_nexthop_1, @ingress::@NoAction_5]
     next_tables [#bmv2ir.action_table<@ingress::@on_miss_3 : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@fib_hit_nexthop_1 : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@NoAction_5 : @ingress::@nexthop_0>]
     type simple
     match_type exact
     keys [#bmv2ir.table_key<type exact, header @ingress1["vrf"]>, #bmv2ir.table_key<type lpm, header @ingress0_ipv4["dstAddr"]>]
     support_timeout false
     default_entry <action @ingress::@NoAction_5, action_const true, action_entry_const true>
     size 16384
    bmv2ir.table @nexthop_0
     actions [@ingress::@on_miss_4, @ingress::@set_egress_details, @ingress::@NoAction_6]
     next_tables [#bmv2ir.action_table<@ingress::@on_miss_4>, #bmv2ir.action_table<@ingress::@set_egress_details>, #bmv2ir.action_table<@ingress::@NoAction_6>]
     type simple
     match_type exact
     keys [#bmv2ir.table_key<type exact, header @ingress1["nexthop_index"]>]
     support_timeout false
     default_entry <action @ingress::@NoAction_6, action_const true, action_entry_const true>
     size 32768
    bmv2ir.table @port_mapping_0
     actions [@ingress::@set_bd, @ingress::@NoAction_7]
     next_tables [#bmv2ir.action_table<@ingress::@set_bd : @ingress::@bd_1>, #bmv2ir.action_table<@ingress::@NoAction_7 : @ingress::@bd_1>]
     type simple
     match_type exact
     keys [#bmv2ir.table_key<type exact, header @ingress2["ingress_port"]>]
     support_timeout false
     default_entry <action @ingress::@NoAction_7, action_const true, action_entry_const true>
     size 32768
    bmv2ir.conditional @conditional_name0 then @ingress::@port_mapping_0 expr {
      %0 = bmv2ir.field @ingress0_ipv4["$valid$"] -> !b1i
      %1 = bmv2ir.d2b %0 : !b1i
      bmv2ir.yield %1 : !p4hir.bool
    }
  }
  bmv2ir.pipeline @egress {
  }
}

// CHECK:  "pipelines": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "conditionals": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "expression": {
// CHECK-NEXT:            "type": "expression",
// CHECK-NEXT:            "value": {
// CHECK-NEXT:              "left": null,
// CHECK-NEXT:              "op": "d2b",
// CHECK-NEXT:              "right": {
// CHECK-NEXT:                "type": "field",
// CHECK-NEXT:                "value": [
// CHECK-NEXT:                  "ingress0_ipv4",
// CHECK-NEXT:                  "$valid$"
// CHECK-NEXT:                ]
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          },
// CHECK-NEXT:          "false_next": null,
// CHECK-NEXT:          "id": 0,
// CHECK-NEXT:          "name": "conditional_name0",
// CHECK-NEXT:          "true_next": "port_mapping_0"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "init_table": "conditional_name0",
// CHECK-NEXT:      "name": "ingress",
// CHECK-NEXT:      "tables": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "action_ids": [
// CHECK-NEXT:            5,
// CHECK-NEXT:            0
// CHECK-NEXT:          ],
// CHECK-NEXT:          "actions": [
// CHECK-NEXT:            "set_vrf",
// CHECK-NEXT:            "NoAction_3"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "default_entry": {
// CHECK-NEXT:            "action_const": true,
// CHECK-NEXT:            "action_data": [],
// CHECK-NEXT:            "action_entry_const": true,
// CHECK-NEXT:            "action_id": 0
// CHECK-NEXT:          },
// CHECK-NEXT:          "id": 0,
// CHECK-NEXT:          "key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "name": "foo.bar"
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress1",
// CHECK-NEXT:                "bd"
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "match_type": "exact",
// CHECK-NEXT:          "max_size": 65536,
// CHECK-NEXT:          "name": "bd_1",
// CHECK-NEXT:          "next_tables": {
// CHECK-NEXT:            "NoAction_3": "ipv4_fib_0",
// CHECK-NEXT:            "set_vrf": "ipv4_fib_0"
// CHECK-NEXT:          },
// CHECK-NEXT:          "support_timeout": false,
// CHECK-NEXT:          "type": "simple"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "action_ids": [
// CHECK-NEXT:            6,
// CHECK-NEXT:            9,
// CHECK-NEXT:            1
// CHECK-NEXT:          ],
// CHECK-NEXT:          "actions": [
// CHECK-NEXT:            "on_miss_2",
// CHECK-NEXT:            "fib_hit_nexthop",
// CHECK-NEXT:            "NoAction_4"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "default_entry": {
// CHECK-NEXT:            "action_const": true,
// CHECK-NEXT:            "action_data": [],
// CHECK-NEXT:            "action_entry_const": true,
// CHECK-NEXT:            "action_id": 1
// CHECK-NEXT:          },
// CHECK-NEXT:          "id": 1,
// CHECK-NEXT:          "key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress1",
// CHECK-NEXT:                "vrf"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress0_ipv4",
// CHECK-NEXT:                "dstAddr"
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "match_type": "exact",
// CHECK-NEXT:          "max_size": 131072,
// CHECK-NEXT:          "name": "ipv4_fib_0",
// CHECK-NEXT:          "next_tables": {
// CHECK-NEXT:            "NoAction_4": "nexthop_0",
// CHECK-NEXT:            "fib_hit_nexthop": "nexthop_0",
// CHECK-NEXT:            "on_miss_2": "ipv4_fib_lpm_0"
// CHECK-NEXT:          },
// CHECK-NEXT:          "support_timeout": false,
// CHECK-NEXT:          "type": "simple"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "action_ids": [
// CHECK-NEXT:            7,
// CHECK-NEXT:            10,
// CHECK-NEXT:            2
// CHECK-NEXT:          ],
// CHECK-NEXT:          "actions": [
// CHECK-NEXT:            "on_miss_3",
// CHECK-NEXT:            "fib_hit_nexthop_1",
// CHECK-NEXT:            "NoAction_5"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "default_entry": {
// CHECK-NEXT:            "action_const": true,
// CHECK-NEXT:            "action_data": [],
// CHECK-NEXT:            "action_entry_const": true,
// CHECK-NEXT:            "action_id": 2
// CHECK-NEXT:          },
// CHECK-NEXT:          "id": 2,
// CHECK-NEXT:          "key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress1",
// CHECK-NEXT:                "vrf"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "lpm",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress0_ipv4",
// CHECK-NEXT:                "dstAddr"
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "match_type": "exact",
// CHECK-NEXT:          "max_size": 16384,
// CHECK-NEXT:          "name": "ipv4_fib_lpm_0",
// CHECK-NEXT:          "next_tables": {
// CHECK-NEXT:            "NoAction_5": "nexthop_0",
// CHECK-NEXT:            "fib_hit_nexthop_1": "nexthop_0",
// CHECK-NEXT:            "on_miss_3": "nexthop_0"
// CHECK-NEXT:          },
// CHECK-NEXT:          "support_timeout": false,
// CHECK-NEXT:          "type": "simple"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "action_ids": [
// CHECK-NEXT:            8,
// CHECK-NEXT:            11,
// CHECK-NEXT:            3
// CHECK-NEXT:          ],
// CHECK-NEXT:          "actions": [
// CHECK-NEXT:            "on_miss_4",
// CHECK-NEXT:            "set_egress_details",
// CHECK-NEXT:            "NoAction_6"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "default_entry": {
// CHECK-NEXT:            "action_const": true,
// CHECK-NEXT:            "action_data": [],
// CHECK-NEXT:            "action_entry_const": true,
// CHECK-NEXT:            "action_id": 3
// CHECK-NEXT:          },
// CHECK-NEXT:          "id": 3,
// CHECK-NEXT:          "key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress1",
// CHECK-NEXT:                "nexthop_index"
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "match_type": "exact",
// CHECK-NEXT:          "max_size": 32768,
// CHECK-NEXT:          "name": "nexthop_0",
// CHECK-NEXT:          "next_tables": {
// CHECK-NEXT:            "NoAction_6": null,
// CHECK-NEXT:            "on_miss_4": null,
// CHECK-NEXT:            "set_egress_details": null
// CHECK-NEXT:          },
// CHECK-NEXT:          "support_timeout": false,
// CHECK-NEXT:          "type": "simple"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "action_ids": [
// CHECK-NEXT:            12,
// CHECK-NEXT:            4
// CHECK-NEXT:          ],
// CHECK-NEXT:          "actions": [
// CHECK-NEXT:            "set_bd",
// CHECK-NEXT:            "NoAction_7"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "default_entry": {
// CHECK-NEXT:            "action_const": true,
// CHECK-NEXT:            "action_data": [],
// CHECK-NEXT:            "action_entry_const": true,
// CHECK-NEXT:            "action_id": 4
// CHECK-NEXT:          },
// CHECK-NEXT:          "id": 4,
// CHECK-NEXT:          "key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "match_type": "exact",
// CHECK-NEXT:              "target": [
// CHECK-NEXT:                "ingress2",
// CHECK-NEXT:                "ingress_port"
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "match_type": "exact",
// CHECK-NEXT:          "max_size": 32768,
// CHECK-NEXT:          "name": "port_mapping_0",
// CHECK-NEXT:          "next_tables": {
// CHECK-NEXT:            "NoAction_7": "bd_1",
// CHECK-NEXT:            "set_bd": "bd_1"
// CHECK-NEXT:          },
// CHECK-NEXT:          "support_timeout": false,
// CHECK-NEXT:          "type": "simple"
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "conditionals": [],
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "init_table": null,
// CHECK-NEXT:      "name": "egress",
// CHECK-NEXT:      "tables": []
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
