// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir,canonicalize)' %s --split-input-file | FileCheck %s
!anon = !p4hir.enum<on_miss, rewrite_src_dst_mac, NoAction_2>
!b12i = !p4hir.bit<12>
!b13i = !p4hir.bit<13>
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!validity_bit = !p4hir.validity.bit
#exact = #p4hir.match_kind<"exact">
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i, _padding: !b4i>
!ingress_metadata_t1 = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
!rewrite_mac_0 = !p4hir.struct<"rewrite_mac_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int32768_infint = #p4hir.int<32768> : !infint
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
module {
  bmv2ir.header_instance @egress0_ethernet : !p4hir.ref<!ethernet_t>
// CHECK:  bmv2ir.header_instance @egress0_ethernet : !bmv2ir.header<"ethernet_t", [dstAddr:!p4hir.bit<48>, srcAddr:!p4hir.bit<48>, etherType:!p4hir.bit<16>], max_length = 14>
    p4hir.func action @rewrite_src_dst_mac(%arg3: !b48i {p4hir.annotations = {name = "smac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "smac"}, %arg4: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "dmac"}) annotations {name = "egress.rewrite_src_dst_mac"} {
      %egress0_ethernet = bmv2ir.symbol_ref @egress0_ethernet : !p4hir.ref<!ethernet_t>
      %srcAddr_field_ref = p4hir.struct_field_ref %egress0_ethernet["srcAddr"] : <!ethernet_t>
      p4hir.assign %arg3, %srcAddr_field_ref : <!b48i>
      %egress0_ethernet_0 = bmv2ir.symbol_ref @egress0_ethernet : !p4hir.ref<!ethernet_t>
      %dstAddr_field_ref = p4hir.struct_field_ref %egress0_ethernet_0["dstAddr"] : <!ethernet_t>
      p4hir.assign %arg4, %dstAddr_field_ref : <!b48i>
      p4hir.return
    }
// CHECK: p4hir.func action @rewrite_src_dst_mac(%arg0: !b48i {p4hir.annotations = {name = "smac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "smac"}, %arg1: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "dmac"}) annotations {name = "egress.rewrite_src_dst_mac"} {
// CHECK:  %0 = bmv2ir.field @egress0_ethernet["srcAddr"] -> !b48i
// CHECK:  bmv2ir.assign %arg0 : !b48i to %0 : !b48i
// CHECK:  %1 = bmv2ir.field @egress0_ethernet["dstAddr"] -> !b48i
// CHECK:  bmv2ir.assign %arg1 : !b48i to %1 : !b48i
// CHECK:  p4hir.return
// CHECK: }


}
