// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s

!b48i = !p4hir.bit<48>
module {
  bmv2ir.header_instance @egress0_ethernet : !bmv2ir.header<"ethernet_t", [dstAddr:!p4hir.bit<48>, srcAddr:!p4hir.bit<48>, etherType:!p4hir.bit<16>], max_length = 14>
  p4hir.func action @rewrite_src_dst_mac(%arg0: !b48i {p4hir.annotations = {name = "smac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "smac"}, %arg1: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "dmac"}) annotations {name = "egress.rewrite_src_dst_mac"} {
    %0 = bmv2ir.field @egress0_ethernet["srcAddr"] -> !b48i
    bmv2ir.assign %arg0 : !b48i to %0 : !b48i
    %1 = bmv2ir.field @egress0_ethernet["dstAddr"] -> !b48i
    bmv2ir.assign %arg1 : !b48i to %1 : !b48i
    p4hir.return
  }
}
// CHECK: "actions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "id": 0,
// CHECK-NEXT:       "name": "rewrite_src_dst_mac",
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
// CHECK-NEXT:   ]
