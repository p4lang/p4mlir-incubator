// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir,canonicalize)' %s --split-input-file | FileCheck %s
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
!type_O = !p4hir.type_var<"O">
!type_T = !p4hir.type_var<"T">
!HashAlgorithm = !p4hir.enum<"HashAlgorithm", crc32, crc32_custom, crc16, crc16_custom, random, identity, csum16, xor16>
#HashAlgorithm_csum16 = #p4hir.enum_field<csum16, !HashAlgorithm> : !HashAlgorithm
#true = #p4hir.bool<true> : !p4hir.bool
!packet_in = !p4hir.extern<"packet_in">
!packet_out = !p4hir.extern<"packet_out">
#undir = #p4hir<dir undir>
#out = #p4hir<dir out>
#inout = #p4hir<dir inout>
#in = #p4hir<dir in>
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
  bmv2ir.header_instance @deparser0_ipv4 : !p4hir.ref<!ipv4_t>
  bmv2ir.header_instance @deparser0_ethernet : !p4hir.ref<!ethernet_t>
  p4hir.parser @p(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"})() {
    p4hir.state @start {
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
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
// CHECK:    bmv2ir.table @bd_1
// CHECK-NEXT:     actions [@ingress::@set_vrf, @ingress::@NoAction_3]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@set_vrf : @ingress::@ipv4_fib_0>, #bmv2ir.action_table<@ingress::@NoAction_3 : @ingress::@ipv4_fib_0>]
// CHECK-NEXT:     type simple
// CHECK-NEXT:     match_type exact
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @ingress1["bd"]>]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@NoAction_3, action_const true, action_entry_const true>
// CHECK-NEXT:     size 65536
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

// CHECK:    bmv2ir.table @ipv4_fib_0
// CHECK-NEXT:     actions [@ingress::@on_miss_2, @ingress::@fib_hit_nexthop, @ingress::@NoAction_4]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@on_miss_2 : @ingress::@ipv4_fib_lpm_0>, #bmv2ir.action_table<@ingress::@fib_hit_nexthop : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@NoAction_4 : @ingress::@nexthop_0>]
// CHECK-NEXT:     type simple
// CHECK-NEXT:     match_type exact
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @ingress1["vrf"]>, #bmv2ir.table_key<type exact, header @ingress0_ipv4["dstAddr"]>]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@NoAction_4, action_const true, action_entry_const true>
// CHECK-NEXT:     size 131072
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
// CHECK:    bmv2ir.table @ipv4_fib_lpm_0
// CHECK-NEXT:     actions [@ingress::@on_miss_3, @ingress::@fib_hit_nexthop_1, @ingress::@NoAction_5]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@on_miss_3 : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@fib_hit_nexthop_1 : @ingress::@nexthop_0>, #bmv2ir.action_table<@ingress::@NoAction_5 : @ingress::@nexthop_0>]
// CHECK-NEXT:     type simple
// CHECK-NEXT:     match_type lpm
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @ingress1["vrf"]>, #bmv2ir.table_key<type lpm, header @ingress0_ipv4["dstAddr"]>]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@NoAction_5, action_const true, action_entry_const true>
// CHECK-NEXT:     size 16384
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
// CHECK:    bmv2ir.table @nexthop_0
// CHECK-NEXT:     actions [@ingress::@on_miss_4, @ingress::@set_egress_details, @ingress::@NoAction_6]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@on_miss_4>, #bmv2ir.action_table<@ingress::@set_egress_details>, #bmv2ir.action_table<@ingress::@NoAction_6>]
// CHECK-NEXT:     type simple
// CHECK-NEXT:     match_type exact
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @ingress1["nexthop_index"]>]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@NoAction_6, action_const true, action_entry_const true>
// CHECK-NEXT:     size 32768
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
// CHECK:    bmv2ir.table @port_mapping_0
// CHECK-NEXT:     actions [@ingress::@set_bd, @ingress::@NoAction_7]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@set_bd : @ingress::@bd_1>, #bmv2ir.action_table<@ingress::@NoAction_7 : @ingress::@bd_1>]
// CHECK-NEXT:     type simple
// CHECK-NEXT:     match_type exact
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @ingress2["ingress_port"]>]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@NoAction_7, action_const true, action_entry_const true>
// CHECK-NEXT:     size 32768
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
// CHECK:    bmv2ir.conditional @conditional_name0 then @ingress::@port_mapping_0 expr {
// CHECK-NEXT:      %0 = bmv2ir.field @ingress0_ipv4["$valid$"] -> !b1i
// CHECK-NEXT:      %1 = bmv2ir.d2b %0 : !b1i
// CHECK-NEXT:      bmv2ir.yield %1 : !p4hir.bool
// CHECK-NEXT:    }
    p4hir.control_apply {
      %ingress0_ipv4 = bmv2ir.symbol_ref @ingress0_ipv4 : !p4hir.ref<!ipv4_t>
      %__valid_field_ref = p4hir.struct_field_ref %ingress0_ipv4["__valid"] : <!ipv4_t>
      %val = p4hir.read %__valid_field_ref : <!validity_bit>
      %eq = p4hir.cmp(eq, %val : !validity_bit, %valid : !validity_bit)
      p4hir.if %eq {
        %port_mapping_0_apply_result = p4hir.table_apply @ingress::@port_mapping_0 with key(%arg2) : (!p4hir.ref<!standard_metadata_t1>) -> !port_mapping_0
        %bd_1_apply_result = p4hir.table_apply @ingress::@bd_1 with key(%arg1) : (!p4hir.ref<!ingress_metadata_t1>) -> !bd_1
        %ipv4_fib_0_apply_result = p4hir.table_apply @ingress::@ipv4_fib_0 with key(%arg1, %arg0) : (!p4hir.ref<!ingress_metadata_t1>, !p4hir.ref<!headers>) -> !ipv4_fib_0
        %action_run = p4hir.struct_extract %ipv4_fib_0_apply_result["action_run"] : !ipv4_fib_0
        p4hir.switch (%action_run : !anon2) {
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
  bmv2ir.header_instance @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
  bmv2ir.header_instance @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
  p4hir.func @verify_checksum<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !type_O {p4hir.dir = #in, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"})
  p4hir.func @update_checksum<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !p4hir.ref<!type_O> {p4hir.dir = #inout, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"}) annotations {pure}
// CHECK:    bmv2ir.calculation @calculation_0 {
// CHECK:    %0 = bmv2ir.field @verifyChecksum0_ipv4["version"] -> !b4i
// CHECK:    %1 = bmv2ir.field @verifyChecksum0_ipv4["ihl"] -> !b4i
// CHECK:    %2 = bmv2ir.field @verifyChecksum0_ipv4["diffserv"] -> !b8i
// CHECK:    %3 = bmv2ir.field @verifyChecksum0_ipv4["totalLen"] -> !b16i
// CHECK:    %4 = bmv2ir.field @verifyChecksum0_ipv4["identification"] -> !b16i
// CHECK:    %5 = bmv2ir.field @verifyChecksum0_ipv4["flags"] -> !b3i
// CHECK:    %6 = bmv2ir.field @verifyChecksum0_ipv4["fragOffset"] -> !b13i
// CHECK:    %7 = bmv2ir.field @verifyChecksum0_ipv4["ttl"] -> !b8i
// CHECK:    %8 = bmv2ir.field @verifyChecksum0_ipv4["protocol"] -> !b8i
// CHECK:    %9 = bmv2ir.field @verifyChecksum0_ipv4["srcAddr"] -> !b32i
// CHECK:    %10 = bmv2ir.field @verifyChecksum0_ipv4["dstAddr"] -> !b32i
// CHECK:    bmv2ir.yield %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : !b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i
// CHECK:  } {algo = "csum16"}
  p4hir.control @verifyChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_apply {
      %verifyChecksum0_ipv4 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val = p4hir.read %verifyChecksum0_ipv4 : <!ipv4_t>
      %version = p4hir.struct_extract %val["version"] : !ipv4_t
      %verifyChecksum0_ipv4_0 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_1 = p4hir.read %verifyChecksum0_ipv4_0 : <!ipv4_t>
      %ihl = p4hir.struct_extract %val_1["ihl"] : !ipv4_t
      %verifyChecksum0_ipv4_2 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_3 = p4hir.read %verifyChecksum0_ipv4_2 : <!ipv4_t>
      %diffserv = p4hir.struct_extract %val_3["diffserv"] : !ipv4_t
      %verifyChecksum0_ipv4_4 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_5 = p4hir.read %verifyChecksum0_ipv4_4 : <!ipv4_t>
      %totalLen = p4hir.struct_extract %val_5["totalLen"] : !ipv4_t
      %verifyChecksum0_ipv4_6 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_7 = p4hir.read %verifyChecksum0_ipv4_6 : <!ipv4_t>
      %identification = p4hir.struct_extract %val_7["identification"] : !ipv4_t
      %verifyChecksum0_ipv4_8 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_9 = p4hir.read %verifyChecksum0_ipv4_8 : <!ipv4_t>
      %flags = p4hir.struct_extract %val_9["flags"] : !ipv4_t
      %verifyChecksum0_ipv4_10 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_11 = p4hir.read %verifyChecksum0_ipv4_10 : <!ipv4_t>
      %fragOffset = p4hir.struct_extract %val_11["fragOffset"] : !ipv4_t
      %verifyChecksum0_ipv4_12 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_13 = p4hir.read %verifyChecksum0_ipv4_12 : <!ipv4_t>
      %ttl = p4hir.struct_extract %val_13["ttl"] : !ipv4_t
      %verifyChecksum0_ipv4_14 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_15 = p4hir.read %verifyChecksum0_ipv4_14 : <!ipv4_t>
      %protocol = p4hir.struct_extract %val_15["protocol"] : !ipv4_t
      %verifyChecksum0_ipv4_16 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_17 = p4hir.read %verifyChecksum0_ipv4_16 : <!ipv4_t>
      %srcAddr = p4hir.struct_extract %val_17["srcAddr"] : !ipv4_t
      %verifyChecksum0_ipv4_18 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_19 = p4hir.read %verifyChecksum0_ipv4_18 : <!ipv4_t>
      %dstAddr = p4hir.struct_extract %val_19["dstAddr"] : !ipv4_t
      %tuple = p4hir.tuple (%version, %ihl, %diffserv, %totalLen, %identification, %flags, %fragOffset, %ttl, %protocol, %srcAddr, %dstAddr) : tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>
      %verifyChecksum0_ipv4_20 = bmv2ir.symbol_ref @verifyChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
      %val_21 = p4hir.read %verifyChecksum0_ipv4_20 : <!ipv4_t>
      %hdrChecksum = p4hir.struct_extract %val_21["hdrChecksum"] : !ipv4_t
      %true = p4hir.const #true
      %HashAlgorithm_csum16 = p4hir.const #HashAlgorithm_csum16
      p4hir.call @verify_checksum<[tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>, !b16i]> (%true, %tuple, %hdrChecksum, %HashAlgorithm_csum16) : (!p4hir.bool, tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>, !b16i, !HashAlgorithm) -> ()
    }
  }
// CHECK:  bmv2ir.calculation @calculation_1 {
// CHECK:    %0 = bmv2ir.field @computeChecksum0_ipv4["version"] -> !b4i
// CHECK:    %1 = bmv2ir.field @computeChecksum0_ipv4["ihl"] -> !b4i
// CHECK:    %2 = bmv2ir.field @computeChecksum0_ipv4["diffserv"] -> !b8i
// CHECK:    %3 = bmv2ir.field @computeChecksum0_ipv4["totalLen"] -> !b16i
// CHECK:    %4 = bmv2ir.field @computeChecksum0_ipv4["identification"] -> !b16i
// CHECK:    %5 = bmv2ir.field @computeChecksum0_ipv4["flags"] -> !b3i
// CHECK:    %6 = bmv2ir.field @computeChecksum0_ipv4["fragOffset"] -> !b13i
// CHECK:    %7 = bmv2ir.field @computeChecksum0_ipv4["ttl"] -> !b8i
// CHECK:    %8 = bmv2ir.field @computeChecksum0_ipv4["protocol"] -> !b8i
// CHECK:    %9 = bmv2ir.field @computeChecksum0_ipv4["srcAddr"] -> !b32i
// CHECK:    %10 = bmv2ir.field @computeChecksum0_ipv4["dstAddr"] -> !b32i
// CHECK:    bmv2ir.yield %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : !b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i
// CHECK:  } {algo = "csum16"}
  p4hir.control @computeChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_apply {
      p4hir.scope {
        %true = p4hir.const #true
        %computeChecksum0_ipv4 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val = p4hir.read %computeChecksum0_ipv4 : <!ipv4_t>
        %version = p4hir.struct_extract %val["version"] : !ipv4_t
        %computeChecksum0_ipv4_0 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_1 = p4hir.read %computeChecksum0_ipv4_0 : <!ipv4_t>
        %ihl = p4hir.struct_extract %val_1["ihl"] : !ipv4_t
        %computeChecksum0_ipv4_2 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_3 = p4hir.read %computeChecksum0_ipv4_2 : <!ipv4_t>
        %diffserv = p4hir.struct_extract %val_3["diffserv"] : !ipv4_t
        %computeChecksum0_ipv4_4 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_5 = p4hir.read %computeChecksum0_ipv4_4 : <!ipv4_t>
        %totalLen = p4hir.struct_extract %val_5["totalLen"] : !ipv4_t
        %computeChecksum0_ipv4_6 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_7 = p4hir.read %computeChecksum0_ipv4_6 : <!ipv4_t>
        %identification = p4hir.struct_extract %val_7["identification"] : !ipv4_t
        %computeChecksum0_ipv4_8 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_9 = p4hir.read %computeChecksum0_ipv4_8 : <!ipv4_t>
        %flags = p4hir.struct_extract %val_9["flags"] : !ipv4_t
        %computeChecksum0_ipv4_10 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_11 = p4hir.read %computeChecksum0_ipv4_10 : <!ipv4_t>
        %fragOffset = p4hir.struct_extract %val_11["fragOffset"] : !ipv4_t
        %computeChecksum0_ipv4_12 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_13 = p4hir.read %computeChecksum0_ipv4_12 : <!ipv4_t>
        %ttl = p4hir.struct_extract %val_13["ttl"] : !ipv4_t
        %computeChecksum0_ipv4_14 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_15 = p4hir.read %computeChecksum0_ipv4_14 : <!ipv4_t>
        %protocol = p4hir.struct_extract %val_15["protocol"] : !ipv4_t
        %computeChecksum0_ipv4_16 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_17 = p4hir.read %computeChecksum0_ipv4_16 : <!ipv4_t>
        %srcAddr = p4hir.struct_extract %val_17["srcAddr"] : !ipv4_t
        %computeChecksum0_ipv4_18 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %val_19 = p4hir.read %computeChecksum0_ipv4_18 : <!ipv4_t>
        %dstAddr = p4hir.struct_extract %val_19["dstAddr"] : !ipv4_t
        %tuple = p4hir.tuple (%version, %ihl, %diffserv, %totalLen, %identification, %flags, %fragOffset, %ttl, %protocol, %srcAddr, %dstAddr) : tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>
        %HashAlgorithm_csum16 = p4hir.const #HashAlgorithm_csum16
        %computeChecksum0_ipv4_20 = bmv2ir.symbol_ref @computeChecksum0_ipv4 : !p4hir.ref<!ipv4_t>
        %hdrChecksum_field_ref = p4hir.struct_field_ref %computeChecksum0_ipv4_20["hdrChecksum"] : <!ipv4_t>
        %checksum_inout_arg = p4hir.variable ["checksum_inout_arg", init] : <!b16i>
        %val_21 = p4hir.read %hdrChecksum_field_ref : <!b16i>
        p4hir.assign %val_21, %checksum_inout_arg : <!b16i>
        p4hir.call @update_checksum<[tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>, !b16i]> (%true, %tuple, %checksum_inout_arg, %HashAlgorithm_csum16) : (!p4hir.bool, tuple<!b4i, !b4i, !b8i, !b16i, !b16i, !b3i, !b13i, !b8i, !b8i, !b32i, !b32i>, !p4hir.ref<!b16i>, !HashAlgorithm) -> ()
        %val_22 = p4hir.read %checksum_inout_arg : <!b16i>
        p4hir.assign %val_22, %hdrChecksum_field_ref : <!b16i>
      }
    }
  }
// CHECK:  bmv2ir.pipeline @egress {
// CHECK:  }
  p4hir.control @egress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"})() {
    p4hir.control_apply {
    }
  }
// CHECK: bmv2ir.deparser @deparser order [@deparser0_ethernet, @deparser0_ipv4]
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !headers {p4hir.dir = #in, p4hir.param_name = "h"})() {
    p4hir.control_apply {
      %ethernet_ref = bmv2ir.symbol_ref @deparser0_ethernet : !p4hir.ref<!ethernet_t>
      %ethernet = p4hir.read %ethernet_ref : <!ethernet_t>
      p4corelib.emit %ethernet : !ethernet_t to %arg0 : !p4corelib.packet_out
      %ipv4_ref = bmv2ir.symbol_ref @deparser0_ipv4 : !p4hir.ref<!ipv4_t>
      %ipv4 = p4hir.read %ipv4_ref : <!ipv4_t>
      p4corelib.emit %ipv4 : !ipv4_t to %arg0 : !p4corelib.packet_out
    }
  }
  bmv2ir.v1switch @main parser @p, verify_checksum @verifyChecksum, ingress @ingress, egress @egress, compute_checksum @computeChecksum, deparser @deparser
}
