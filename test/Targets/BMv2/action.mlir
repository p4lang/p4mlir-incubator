// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s

!b48i = !p4hir.bit<48>
!b16i = !p4hir.bit<16>
#int-1_b16i = #p4hir.int<65535> : !b16i
module {
  bmv2ir.header_instance @egress0_ethernet : !bmv2ir.header<"ethernet_t", [dstAddr:!p4hir.bit<48>, srcAddr:!p4hir.bit<48>, etherType:!p4hir.bit<16>], max_length = 14>
  p4hir.func action @rewrite_src_dst_mac(%arg0: !b48i {p4hir.annotations = {name = "smac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "smac"}, %arg1: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "dmac"}) annotations {name = "egress.rewrite_src_dst_mac"} {
    %0 = bmv2ir.field @egress0_ethernet["srcAddr"] -> !b48i
    bmv2ir.assign %arg0 : !b48i to %0 : !b48i
    %1 = bmv2ir.field @egress0_ethernet["dstAddr"] -> !b48i
    bmv2ir.assign %arg1 : !b48i to %1 : !b48i
    p4hir.return
  }

  p4hir.func action @fib_hit_nexthop(%arg0: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "nexthop_index_1"}) annotations {name = "ingress.fib_hit_nexthop"} {
    %c-1_b8i = p4hir.const #int-1_b16i
    %0 = bmv2ir.field @egress0_ethernet["dstAddr"] -> !b16i
    bmv2ir.assign %arg0 : !b16i to %0 : !b16i
    %1 = bmv2ir.field @egress0_ethernet["etherType"] -> !b16i
    %2 = bmv2ir.field @egress0_ethernet["etherType"] -> !b16i
    %add = p4hir.binop(add, %2, %c-1_b8i) : !b16i
    bmv2ir.assign %add : !b16i to %1 : !b16i
    p4hir.return
  }
}

// CHECK: "actions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "id": 0,
// CHECK-NEXT:       "name": "egress.rewrite_src_dst_mac",
// CHECK-NEXT:       "primitives": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "op": "assign",
// CHECK-NEXT:           "parameters": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "type": "field",
// CHECK-NEXT:               "value": [
// CHECK-NEXT:                 "egress0_ethernet",
// CHECK-NEXT:                 "srcAddr"
// CHECK-NEXT:               ]
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "type": "runtime_data",
// CHECK-NEXT:               "value": 0
// CHECK-NEXT:             }
// CHECK-NEXT:           ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "op": "assign",
// CHECK-NEXT:           "parameters": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "type": "field",
// CHECK-NEXT:               "value": [
// CHECK-NEXT:                 "egress0_ethernet",
// CHECK-NEXT:                 "dstAddr"
// CHECK-NEXT:               ]
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "type": "runtime_data",
// CHECK-NEXT:               "value": 1
// CHECK-NEXT:             }
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "runtime_data": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "bitwidth": 48,
// CHECK-NEXT:           "name": "smac"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "bitwidth": 48,
// CHECK-NEXT:           "name": "dmac"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:    {
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "name": "ingress.fib_hit_nexthop",
// CHECK-NEXT:      "primitives": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "op": "assign",
// CHECK-NEXT:          "parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "field",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                "egress0_ethernet",
// CHECK-NEXT:                "dstAddr"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "runtime_data",
// CHECK-NEXT:              "value": 0
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "op": "assign",
// CHECK-NEXT:          "parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "field",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                "egress0_ethernet",
// CHECK-NEXT:                "etherType"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "expression",
// CHECK-NEXT:              "value": {
// CHECK-NEXT:                 "type": "expression",
// CHECK-NEXT:                 "value": {
// CHECK-NEXT:                   "left": {
// CHECK-NEXT:                     "type": "expression",
// CHECK-NEXT:                     "value": {
// CHECK-NEXT:                       "left": {
// CHECK-NEXT:                         "type": "field",
// CHECK-NEXT:                         "value": [
// CHECK-NEXT:                           "egress0_ethernet",
// CHECK-NEXT:                           "etherType"
// CHECK-NEXT:                         ]
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "op": "+",
// CHECK-NEXT:                       "right": {
// CHECK-NEXT:                         "type": "hexstr",
// CHECK-NEXT:                         "value": "0xffff"
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "op": "&",
// CHECK-NEXT:                   "right": {
// CHECK-NEXT:                     "type": "hexstr",
// CHECK-NEXT:                     "value": "0xffff"
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "runtime_data": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "bitwidth": 16,
// CHECK-NEXT:          "name": "nexthop_index"
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  ],

// -----

!b32i = !p4hir.bit<32>
!b9i = !p4hir.bit<9>
!b64i = !p4hir.bit<64>
#int0_b9i = #p4hir.int<0> : !b9i
module {
    bmv2ir.header_instance @Headers_h : !bmv2ir.header<"hdr", [a:!p4hir.bit<32>, b:!p4hir.bit<32>, c:!p4hir.bit<64>], max_length = 16>
    bmv2ir.header_instance @standard_metadata : !bmv2ir.header<"standard_metadata", [ingress_port:!p4hir.bit<9>, egress_spec:!p4hir.bit<9>, egress_port:!p4hir.bit<9>, instance_type:!p4hir.bit<32>, packet_length:!p4hir.bit<32>, enq_timestamp:!p4hir.bit<32>, enq_qdepth:!p4hir.bit<19>, deq_timedelta:!p4hir.bit<32>, deq_qdepth:!p4hir.bit<19>, ingress_global_timestamp:!p4hir.bit<48>, egress_global_timestamp:!p4hir.bit<48>, mcast_grp:!p4hir.bit<16>, egress_rid:!p4hir.bit<16>, checksum_error:!p4hir.bit<1>, priority:!p4hir.bit<3>, _padding:!p4hir.bit<3>], max_length = 41>
    p4hir.func action @add() annotations {name = "ingress.add"} {
      %c0_b9i = p4hir.const #int0_b9i
      %0 = bmv2ir.field @Headers_h["c"] -> !b64i
      %1 = bmv2ir.field @Headers_h["a"] -> !b32i
      %2 = bmv2ir.field @Headers_h["b"] -> !b32i
      %add = p4hir.binop(add, %1, %2) : !b32i
      %cast = p4hir.cast(%add : !b32i) : !b64i
      bmv2ir.assign %cast : !b64i to %0 : !b64i
      %3 = bmv2ir.field @standard_metadata["egress_spec"] -> !b9i
      bmv2ir.assign %c0_b9i : !b9i to %3 : !b9i
      p4hir.return
    }

}

// CHECK-LABEL:      "name": "ingress.add",
// CHECK-NEXT:      "primitives": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "op": "assign",
// CHECK-NEXT:          "parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "field",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                "Headers_h",
// CHECK-NEXT:                "c"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "expression",
// CHECK-NEXT:              "value": {
// CHECK-NEXT:                "type": "expression",
// CHECK-NEXT:                "value": {
// CHECK-NEXT:                  "left": {
// CHECK-NEXT:                    "type": "expression",
// CHECK-NEXT:                    "value": {
// CHECK-NEXT:                      "left": {
// CHECK-NEXT:                        "type": "expression",
// CHECK-NEXT:                        "value": {
// CHECK-NEXT:                          "left": {
// CHECK-NEXT:                            "type": "field",
// CHECK-NEXT:                            "value": [
// CHECK-NEXT:                              "Headers_h",
// CHECK-NEXT:                              "a"
// CHECK-NEXT:                            ]
// CHECK-NEXT:                          },
// CHECK-NEXT:                          "op": "+",
// CHECK-NEXT:                          "right": {
// CHECK-NEXT:                            "type": "field",
// CHECK-NEXT:                            "value": [
// CHECK-NEXT:                              "Headers_h",
// CHECK-NEXT:                              "b"
// CHECK-NEXT:                            ]
// CHECK-NEXT:                          }
// CHECK-NEXT:                        }
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "op": "&",
// CHECK-NEXT:                      "right": {
// CHECK-NEXT:                        "type": "hexstr",
// CHECK-NEXT:                        "value": "0xffffffff"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "op": "&",
// CHECK-NEXT:                  "right": {
// CHECK-NEXT:                    "type": "hexstr",
// CHECK-NEXT:                    "value": "0xffffffffffffffff"
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "op": "assign",
// CHECK-NEXT:          "parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "field",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                "standard_metadata",
// CHECK-NEXT:                "egress_spec"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "hexstr",
// CHECK-NEXT:              "value": "0x0000"
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "runtime_data": []
// CHECK-NEXT:    }

// -----

!i32i = !p4hir.int<32>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
#int0_b9i = #p4hir.int<0> : !b9i
module {
  bmv2ir.header_instance @standard_metadata : !bmv2ir.header<"standard_metadata", [ingress_port:!p4hir.bit<9>, egress_spec:!p4hir.bit<9>, egress_port:!p4hir.bit<9>, instance_type:!p4hir.bit<32>, packet_length:!p4hir.bit<32>, enq_timestamp:!p4hir.bit<32>, enq_qdepth:!p4hir.bit<19>, deq_timedelta:!p4hir.bit<32>, deq_qdepth:!p4hir.bit<19>, ingress_global_timestamp:!p4hir.bit<48>, egress_global_timestamp:!p4hir.bit<48>, mcast_grp:!p4hir.bit<16>, egress_rid:!p4hir.bit<16>, checksum_error:!p4hir.bit<1>, priority:!p4hir.bit<3>, _padding:!p4hir.bit<3>], max_length = 41>
  bmv2ir.header_instance @Headers_h : !bmv2ir.header<"hdr", [a:!p4hir.int<32>, b:!p4hir.int<32>, c:!p4hir.bit<8>], max_length = 9>

    p4hir.func action @compare() annotations {name = "ingress.compare"} {
      %c0_b9i = p4hir.const #int0_b9i
      %0 = bmv2ir.field @Headers_h["c"] -> !b8i
      %1 = bmv2ir.field @Headers_h["a"] -> !i32i
      %2 = bmv2ir.field @Headers_h["b"] -> !i32i
      %lt = p4hir.cmp(lt, %1 : !i32i, %2 : !i32i)
      %cast = p4hir.cast(%lt : !p4hir.bool) : !b8i
      bmv2ir.assign %cast : !b8i to %0 : !b8i
      %3 = bmv2ir.field @standard_metadata["egress_spec"] -> !b9i
      bmv2ir.assign %c0_b9i : !b9i to %3 : !b9i
      p4hir.return
    }
}


// CHECK-LABEL:      "name": "ingress.compare",
// CHECK-NEXT:      "primitives": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "op": "assign",
// CHECK-NEXT:          "parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "field",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                "Headers_h",
// CHECK-NEXT:                "c"
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "expression",
// CHECK-NEXT:              "value": {
// CHECK-NEXT:                "type": "expression",
// CHECK-NEXT:                "value": {
// CHECK-NEXT:                  "left": {
// CHECK-NEXT:                    "type": "expression",
// CHECK-NEXT:                    "value": {
// CHECK-NEXT:                      "cond": {
// CHECK-NEXT:                        "type": "expression",
// CHECK-NEXT:                        "value": {
// CHECK-NEXT:                          "left": {
// CHECK-NEXT:                            "type": "field",
// CHECK-NEXT:                            "value": [
// CHECK-NEXT:                              "Headers_h",
// CHECK-NEXT:                              "a"
// CHECK-NEXT:                            ]
// CHECK-NEXT:                          },
// CHECK-NEXT:                          "op": "<",
// CHECK-NEXT:                          "right": {
// CHECK-NEXT:                            "type": "field",
// CHECK-NEXT:                            "value": [
// CHECK-NEXT:                              "Headers_h",
// CHECK-NEXT:                              "b"
// CHECK-NEXT:                            ]
// CHECK-NEXT:                          }
// CHECK-NEXT:                        }
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "left": {
// CHECK-NEXT:                        "type": "hexstr",
// CHECK-NEXT:                        "value": "0x01"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "op": "?",
// CHECK-NEXT:                      "right": {
// CHECK-NEXT:                        "type": "hexstr",
// CHECK-NEXT:                        "value": "0x00"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "op": "&",
// CHECK-NEXT:                  "right": {
// CHECK-NEXT:                    "type": "hexstr",
// CHECK-NEXT:                    "value": "0xff"
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
