// RUN: p4mlir-opt -p='builtin.module(synth-actions)' %s --split-input-file | FileCheck %s
!Meta = !p4hir.struct<"Meta">
!MeterType = !p4hir.enum<"MeterType", packets, bytes>
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!packet_out = !p4hir.extern<"packet_out">
!string = !p4hir.string
!type_D = !p4hir.type_var<"D">
!type_H = !p4hir.type_var<"H">
!type_M = !p4hir.type_var<"M">
!type_O = !p4hir.type_var<"O">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#inout = #p4hir<dir inout>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!H = !p4hir.header<"H", a: !b8i, b: !b8i, __valid: !validity_bit>
!ethernet_t = !p4hir.header<"ethernet_t", dst_addr: !b48i, src_addr: !b48i, eth_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int-1_b16i = #p4hir.int<65535> : !b16i
#int-1_b48i = #p4hir.int<281474976710655> : !b48i
#int10_b8i = #p4hir.int<10> : !b8i
#int11_b8i = #p4hir.int<11> : !b8i
#int255_b9i = #p4hir.int<255> : !b9i
#valid = #p4hir<validity.bit valid> : !validity_bit
!headers = !p4hir.struct<"headers", eth_hdr: !ethernet_t, h: !H>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
!ingress = !p4hir.control<"ingress", (!p4hir.ref<!headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t1>)>
// CHECK: #[[$ATTR_0:.+]] = #p4hir.int<65535> : !b16i
// CHECK: #[[$ATTR_1:.+]] = #p4hir.int<281474976710655> : !b48i
// CHECK: #[[$ATTR_2:.+]] = #p4hir.int<10> : !b8i
// CHECK: #[[$ATTR_3:.+]] = #p4hir.int<11> : !b8i
// CHECK: #[[$ATTR_4:.+]] = #p4hir<validity.bit valid> : !validity_bit
module {
  bmv2ir.header_instance @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
  bmv2ir.header_instance @ingress0_h : !p4hir.ref<!H>
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
// CHECK:           p4hir.func action @dummy_action_2() {
// CHECK:             %[[VAL_0:.*]] = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
// CHECK:             %[[VAL_1:.*]] = p4hir.struct_field_ref %[[VAL_0]]["src_addr"] : <!ethernet_t>
// CHECK:             %[[VAL_2:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.assign %[[VAL_2]], %[[VAL_1]] : <!b48i>
// CHECK:             p4hir.return
// CHECK:           }

// CHECK:           p4hir.func action @dummy_action_1() {
// CHECK:             %[[VAL_0:.*]] = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
// CHECK:             %[[VAL_1:.*]] = p4hir.struct_field_ref %[[VAL_0]]["dst_addr"] : <!ethernet_t>
// CHECK:             %[[VAL_2:.*]] = p4hir.const #[[$ATTR_1]]
// CHECK:             p4hir.assign %[[VAL_2]], %[[VAL_1]] : <!b48i>
// CHECK:             %[[VAL_3:.*]] = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
// CHECK:             %[[VAL_4:.*]] = p4hir.struct_field_ref %[[VAL_3]]["src_addr"] : <!ethernet_t>
// CHECK:             p4hir.assign %[[VAL_2]], %[[VAL_4]] : <!b48i>
// CHECK:             %[[VAL_5:.*]] = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
// CHECK:             %[[VAL_6:.*]] = p4hir.struct_field_ref %[[VAL_5]]["eth_type"] : <!ethernet_t>
// CHECK:             %[[VAL_7:.*]] = p4hir.const #[[$ATTR_0]]
// CHECK:             p4hir.assign %[[VAL_7]], %[[VAL_6]] : <!b16i>
// CHECK:             p4hir.return
// CHECK:           }

// CHECK:           p4hir.func action @dummy_action_0() {
// CHECK:             %[[VAL_8:.*]] = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
// CHECK:             %[[VAL_9:.*]] = p4hir.struct_field_ref %[[VAL_8]]["a"] : <!H>
// CHECK:             %[[VAL_10:.*]] = p4hir.const #[[$ATTR_2]]
// CHECK:             p4hir.assign %[[VAL_10]], %[[VAL_9]] : <!b8i>
// CHECK:             %[[VAL_11:.*]] = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
// CHECK:             %[[VAL_12:.*]] = p4hir.struct_field_ref %[[VAL_11]]["b"] : <!H>
// CHECK:             %[[VAL_13:.*]] = p4hir.const #[[$ATTR_3]]
// CHECK:             p4hir.assign %[[VAL_13]], %[[VAL_12]] : <!b8i>
// CHECK:             p4hir.return
// CHECK:           }
    %c11_b8i = p4hir.const #int11_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %valid = p4hir.const #valid
    %c-1_b16i = p4hir.const #int-1_b16i
    %c-1_b48i = p4hir.const #int-1_b48i
    p4hir.control_apply {
      %ingress0_eth_hdr = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %dst_addr_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr["dst_addr"] : <!ethernet_t>
      p4hir.assign %c-1_b48i, %dst_addr_field_ref : <!b48i>
      %ingress0_eth_hdr_0 = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %src_addr_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr_0["src_addr"] : <!ethernet_t>
      p4hir.assign %c-1_b48i, %src_addr_field_ref : <!b48i>
      %ingress0_eth_hdr_1 = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %eth_type_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr_1["eth_type"] : <!ethernet_t>
      p4hir.assign %c-1_b16i, %eth_type_field_ref : <!b16i>
      %ingress0_h = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
      %__valid_field_ref = p4hir.struct_field_ref %ingress0_h["__valid"] : <!H>
      %val = p4hir.read %__valid_field_ref : <!validity_bit>
      %eq = p4hir.cmp(eq, %val : !validity_bit, %valid : !validity_bit)
      p4hir.if %eq {
        %ingress0_h_2 = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
        %a_field_ref = p4hir.struct_field_ref %ingress0_h_2["a"] : <!H>
        p4hir.assign %c10_b8i, %a_field_ref : <!b8i>
        %ingress0_h_3 = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
        %b_field_ref = p4hir.struct_field_ref %ingress0_h_3["b"] : <!H>
        p4hir.assign %c11_b8i, %b_field_ref : <!b8i>
      }
      %ingress0_eth_hdr_0_1 = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %src_addr_field_ref_1 = p4hir.struct_field_ref %ingress0_eth_hdr_0_1["src_addr"] : <!ethernet_t>
      p4hir.assign %c-1_b48i, %src_addr_field_ref_1 : <!b48i>
    }
  }
// CHECK:           p4hir.control_apply {
// CHECK:             p4hir.call @ingress::@dummy_action_1 () : () -> ()
// CHECK:             bmv2ir.if @cond_node_0 expr {
// CHECK:               %[[VAL_18:.*]] = p4hir.const #[[$ATTR_4]]
// CHECK:               %[[VAL_19:.*]] = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
// CHECK:               %[[VAL_20:.*]] = p4hir.struct_field_ref %[[VAL_19]]["__valid"] : <!H>
// CHECK:               %[[VAL_21:.*]] = p4hir.read %[[VAL_20]] : <!validity_bit>
// CHECK:               %[[VAL_22:.*]] = p4hir.cmp(eq, %[[VAL_21]] : !validity_bit, %[[VAL_18]] : !validity_bit)
// CHECK:               bmv2ir.yield %[[VAL_22]] : !p4hir.bool
// CHECK:             } then {
// CHECK:               p4hir.call @ingress::@dummy_action_0 () : () -> ()
// CHECK:             } else {
// CHECK:             }
// CHECK:             p4hir.call @ingress::@dummy_action_2 () : () -> ()
// CHECK:           }
  p4hir.parser @p()() {
    p4hir.state @start {
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @computeChecksum()() {
    p4hir.control_apply {
    }
  }
  p4hir.control @verifyChecksum()() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser()() {
    p4hir.control_apply {
    }
  }

  p4hir.control @egress()() {
    p4hir.control_apply {
    }
  }
  bmv2ir.v1switch @main parser @p, verify_checksum @verifyChecksum, ingress @ingress, egress @egress, compute_checksum @computeChecksum, deparser @deparser
}

// -----

// Checks that we correcly handle switch cases after table_apply
// TODO: narrow down test case

!anon = !p4hir.enum<set_bd, NoAction_7>
!anon1 = !p4hir.enum<set_vrf, NoAction_3>
!anon2 = !p4hir.enum<on_miss_2, fib_hit_nexthop, NoAction_4>
!anon3 = !p4hir.enum<on_miss_3, fib_hit_nexthop_1, NoAction_5>
!anon4 = !p4hir.enum<on_miss_4, set_egress_details, NoAction_6>
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
#lpm = #p4hir.match_kind<"lpm">
!bd_1 = !p4hir.struct<"bd_1", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon1>
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i, _padding: !b4i>
!ingress_metadata_t1 = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i>
!ipv4_fib_0 = !p4hir.struct<"ipv4_fib_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon2>
!ipv4_fib_lpm_0 = !p4hir.struct<"ipv4_fib_lpm_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon3>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
!nexthop_0 = !p4hir.struct<"nexthop_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon4>
!port_mapping_0 = !p4hir.struct<"port_mapping_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#anon_on_miss_2 = #p4hir.enum_field<on_miss_2, !anon2> : !anon2
#int-1_b8i = #p4hir.int<255> : !b8i
#int131072_infint = #p4hir.int<131072> : !infint
#int16384_infint = #p4hir.int<16384> : !infint
#int32768_infint = #p4hir.int<32768> : !infint
#int65536_infint = #p4hir.int<65536> : !infint
#valid = #p4hir<validity.bit valid> : !validity_bit
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
module {
  bmv2ir.header_instance @ingress2 : !p4hir.ref<!standard_metadata_t>
  bmv2ir.header_instance @ingress1 : !p4hir.ref<!ingress_metadata_t>
  bmv2ir.header_instance @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!ingress_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_metadata"})() {
    %valid = p4hir.const #valid
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
    p4hir.func action @set_vrf(%arg3: !b12i {p4hir.annotations = {name = "vrf"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "vrf_1"}) annotations {name = "ingress.set_vrf"} {
      %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
      %vrf_field_ref = p4hir.struct_field_ref %ingress1["vrf"] : <!ingress_metadata_t>
      p4hir.assign %arg3, %vrf_field_ref : <!b12i>
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
    p4hir.func action @fib_hit_nexthop(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "nexthop_index_1"}) annotations {name = "ingress.fib_hit_nexthop"} {
      %c-1_b8i = p4hir.const #int-1_b8i
      %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
      %nexthop_index_field_ref = p4hir.struct_field_ref %ingress1["nexthop_index"] : <!ingress_metadata_t>
      p4hir.assign %arg3, %nexthop_index_field_ref : <!b16i>
      %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %ttl_field_ref = p4hir.struct_field_ref %ingress0_ipv4["ttl"] : <!ipv4_t>
      %ingress0_ipv4_0 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %ttl_field_ref_1 = p4hir.struct_field_ref %ingress0_ipv4_0["ttl"] : <!ipv4_t>
      %val = p4hir.read %ttl_field_ref_1 : <!b8i>
      %add = p4hir.binop(add, %val, %c-1_b8i) : !b8i
      p4hir.assign %add, %ttl_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.func action @fib_hit_nexthop_1(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "nexthop_index_2"}) annotations {name = "ingress.fib_hit_nexthop"} {
      %c-1_b8i = p4hir.const #int-1_b8i
      %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
      %nexthop_index_field_ref = p4hir.struct_field_ref %ingress1["nexthop_index"] : <!ingress_metadata_t>
      p4hir.assign %arg3, %nexthop_index_field_ref : <!b16i>
      %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %ttl_field_ref = p4hir.struct_field_ref %ingress0_ipv4["ttl"] : <!ipv4_t>
      %ingress0_ipv4_0 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %ttl_field_ref_1 = p4hir.struct_field_ref %ingress0_ipv4_0["ttl"] : <!ipv4_t>
      %val = p4hir.read %ttl_field_ref_1 : <!b8i>
      %add = p4hir.binop(add, %val, %c-1_b8i) : !b8i
      p4hir.assign %add, %ttl_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.func action @set_egress_details(%arg3: !b9i {p4hir.annotations = {name = "egress_spec"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "egress_spec_1"}) annotations {name = "ingress.set_egress_details"} {
      %ingress2 = bmv2ir.symbol_ref @ingress2 : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %ingress2["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %arg3, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.func action @set_bd(%arg3: !b16i {p4hir.annotations = {name = "bd"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "bd_0"}) annotations {name = "ingress.set_bd"} {
      %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
      %bd_field_ref = p4hir.struct_field_ref %ingress1["bd"] : <!ingress_metadata_t>
      p4hir.assign %arg3, %bd_field_ref : <!b16i>
      p4hir.return
    }
    p4hir.table @bd_1 annotations {name = "ingress.bd"} {
      p4hir.table_actions {
        p4hir.table_action @set_vrf(%arg3: !b12i {p4hir.annotations = {name = "vrf"}, p4hir.param_name = "vrf_1"}) {
          p4hir.call @ingress::@set_vrf (%arg3) : (!b12i) -> ()
        }
        p4hir.table_action @NoAction_3() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_3 () : () -> ()
        }
      }
      p4hir.table_key(%arg3: !p4hir.ref<!ingress_metadata_t1>) {
        %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
        %bd_field_ref = p4hir.struct_field_ref %ingress1["bd"] : <!ingress_metadata_t>
        %val = p4hir.read %bd_field_ref : <!b16i>
        p4hir.match_key #exact %val : !b16i annotations {name = "meta.ingress_metadata.bd"}
      }
      %size = p4hir.table_size #int65536_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_3 () : () -> ()
      }
    }

    p4hir.table @ipv4_fib_0 annotations {name = "ingress.ipv4_fib"} {
      p4hir.table_actions {
        p4hir.table_action @on_miss_2() {
          p4hir.call @ingress::@on_miss_2 () : () -> ()
        }
        p4hir.table_action @fib_hit_nexthop(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.param_name = "nexthop_index_1"}) {
          p4hir.call @ingress::@fib_hit_nexthop (%arg3) : (!b16i) -> ()
        }
        p4hir.table_action @NoAction_4() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_4 () : () -> ()
        }
      }
      p4hir.table_key(%arg3: !p4hir.ref<!ingress_metadata_t1>, %arg4: !p4hir.ref<!headers>) {
        %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
        %vrf_field_ref = p4hir.struct_field_ref %ingress1["vrf"] : <!ingress_metadata_t>
        %val = p4hir.read %vrf_field_ref : <!b12i>
        p4hir.match_key #exact %val : !b12i annotations {name = "meta.ingress_metadata.vrf"}
        %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
        %dstAddr_field_ref = p4hir.struct_field_ref %ingress0_ipv4["dstAddr"] : <!ipv4_t>
        %val_0 = p4hir.read %dstAddr_field_ref : <!b32i>
        p4hir.match_key #exact %val_0 : !b32i annotations {name = "hdr.ipv4.dstAddr"}
      }
      %size = p4hir.table_size #int131072_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_4 () : () -> ()
      }
    }
    p4hir.table @ipv4_fib_lpm_0 annotations {name = "ingress.ipv4_fib_lpm"} {
      p4hir.table_actions {
        p4hir.table_action @on_miss_3() {
          p4hir.call @ingress::@on_miss_3 () : () -> ()
        }
        p4hir.table_action @fib_hit_nexthop_1(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.param_name = "nexthop_index_2"}) {
          p4hir.call @ingress::@fib_hit_nexthop_1 (%arg3) : (!b16i) -> ()
        }
        p4hir.table_action @NoAction_5() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_5 () : () -> ()
        }
      }
      p4hir.table_key(%arg3: !p4hir.ref<!ingress_metadata_t1>, %arg4: !p4hir.ref<!headers>) {
        %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
        %vrf_field_ref = p4hir.struct_field_ref %ingress1["vrf"] : <!ingress_metadata_t>
        %val = p4hir.read %vrf_field_ref : <!b12i>
        p4hir.match_key #exact %val : !b12i annotations {name = "meta.ingress_metadata.vrf"}
        %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
        %dstAddr_field_ref = p4hir.struct_field_ref %ingress0_ipv4["dstAddr"] : <!ipv4_t>
        %val_0 = p4hir.read %dstAddr_field_ref : <!b32i>
        p4hir.match_key #lpm %val_0 : !b32i annotations {name = "hdr.ipv4.dstAddr"}
      }
      %size = p4hir.table_size #int16384_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_5 () : () -> ()
      }
    }
    p4hir.table @nexthop_0 annotations {name = "ingress.nexthop"} {
      p4hir.table_actions {
        p4hir.table_action @on_miss_4() {
          p4hir.call @ingress::@on_miss_4 () : () -> ()
        }
        p4hir.table_action @set_egress_details(%arg3: !b9i {p4hir.annotations = {name = "egress_spec"}, p4hir.param_name = "egress_spec_1"}) {
          p4hir.call @ingress::@set_egress_details (%arg3) : (!b9i) -> ()
        }
        p4hir.table_action @NoAction_6() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_6 () : () -> ()
        }
      }
      p4hir.table_key(%arg3: !p4hir.ref<!ingress_metadata_t1>) {
        %ingress1 = bmv2ir.symbol_ref @ingress1 : !p4hir.ref<!ingress_metadata_t>
        %nexthop_index_field_ref = p4hir.struct_field_ref %ingress1["nexthop_index"] : <!ingress_metadata_t>
        %val = p4hir.read %nexthop_index_field_ref : <!b16i>
        p4hir.match_key #exact %val : !b16i annotations {name = "meta.ingress_metadata.nexthop_index"}
      }
      %size = p4hir.table_size #int32768_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_6 () : () -> ()
      }
    }
    p4hir.table @port_mapping_0 annotations {name = "ingress.port_mapping"} {
      p4hir.table_actions {
        p4hir.table_action @set_bd(%arg3: !b16i {p4hir.annotations = {name = "bd"}, p4hir.param_name = "bd_0"}) {
          p4hir.call @ingress::@set_bd (%arg3) : (!b16i) -> ()
        }
        p4hir.table_action @NoAction_7() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_7 () : () -> ()
        }
      }
      p4hir.table_key(%arg3: !p4hir.ref<!standard_metadata_t1>) {
        %ingress2 = bmv2ir.symbol_ref @ingress2 : !p4hir.ref<!standard_metadata_t>
        %ingress_port_field_ref = p4hir.struct_field_ref %ingress2["ingress_port"] : <!standard_metadata_t>
        %val = p4hir.read %ingress_port_field_ref : <!b9i>
        p4hir.match_key #exact %val : !b9i annotations {name = "standard_metadata.ingress_port"}
      }
      %size = p4hir.table_size #int32768_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_7 () : () -> ()
      }
    }
    p4hir.control_apply {
      %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %__valid_field_ref = p4hir.struct_field_ref %ingress0_ipv4["__valid"] : <!ipv4_t>
      %val = p4hir.read %__valid_field_ref : <!validity_bit>
      %eq = p4hir.cmp(eq, %val : !validity_bit, %valid : !validity_bit)
// CHECK: bmv2ir.if @cond_node_0 expr
      p4hir.if %eq {
        %port_mapping_0_apply_result = p4hir.table_apply @ingress::@port_mapping_0 with key(%arg2) : (!p4hir.ref<!standard_metadata_t1>) -> !port_mapping_0
        %bd_1_apply_result = p4hir.table_apply @ingress::@bd_1 with key(%arg1) : (!p4hir.ref<!ingress_metadata_t1>) -> !bd_1
        %ipv4_fib_0_apply_result = p4hir.table_apply @ingress::@ipv4_fib_0 with key(%arg1, %arg0) : (!p4hir.ref<!ingress_metadata_t1>, !p4hir.ref<!headers>) -> !ipv4_fib_0
        %action_run = p4hir.struct_extract %ipv4_fib_0_apply_result["action_run"] : !ipv4_fib_0
        p4hir.switch (%action_run : !anon2) {
// CHECK:  p4hir.switch (%{{.*}} : !anon2)
          p4hir.case(equal, [#anon_on_miss_2]) {
            %ipv4_fib_lpm_0_apply_result = p4hir.table_apply @ingress::@ipv4_fib_lpm_0 with key(%arg1, %arg0) : (!p4hir.ref<!ingress_metadata_t1>, !p4hir.ref<!headers>) -> !ipv4_fib_lpm_0
            p4hir.yield
          }
          p4hir.case(default, []) {
            p4hir.yield
          }
          p4hir.yield
        }
        %nexthop_0_apply_result = p4hir.table_apply @ingress::@nexthop_0 with key(%arg1) : (!p4hir.ref<!ingress_metadata_t1>) -> !nexthop_0
      }
    }
  }
  p4hir.parser @p()() {
    p4hir.state @start {
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @computeChecksum()() {
    p4hir.control_apply {
    }
  }
  p4hir.control @verifyChecksum()() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser()() {
    p4hir.control_apply {
    }
  }

  p4hir.control @egress()() {
    p4hir.control_apply {
    }
  }
  bmv2ir.v1switch @main parser @p, verify_checksum @verifyChecksum, ingress @ingress, egress @egress, compute_checksum @computeChecksum, deparser @deparser
}
