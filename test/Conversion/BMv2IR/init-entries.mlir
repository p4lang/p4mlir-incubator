// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir,canonicalize)' %s --split-input-file | FileCheck %s
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!metadata_t = !p4hir.struct<"metadata_t">
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#exact = #p4hir.match_kind<"exact">
#false = #p4hir.bool<false> : !p4hir.bool
#ternary = #p4hir.match_kind<"ternary">
!ethernet_h = !p4hir.header<"ethernet_h", dst_addr: !b48i, src_addr: !b48i, ether_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t1_0 = !p4hir.struct<"t1_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !b32i>
#int-4096_b16i = #p4hir.int<61440> : !b16i
#int10_infint = #p4hir.int<10> : !infint
#int15_b16i = #p4hir.int<15> : !b16i
#int16_b16i = #p4hir.int<16> : !b16i
#int1_b32i = #p4hir.int<1> : !b32i
#int1_b48i = #p4hir.int<1> : !b48i
#int20_infint = #p4hir.int<20> : !infint
#int2_b32i = #p4hir.int<2> : !b32i
#int2_b48i = #p4hir.int<2> : !b48i
#int30_infint = #p4hir.int<30> : !infint
#int3_b32i = #p4hir.int<3> : !b32i
#int3_b48i = #p4hir.int<3> : !b48i
#int4096_b16i = #p4hir.int<4096> : !b16i
#int40_infint = #p4hir.int<40> : !infint
#int4369_b16i = #p4hir.int<4369> : !b16i
#int4481_b16i = #p4hir.int<4481> : !b16i
#int4_b32i = #p4hir.int<4> : !b32i
#int4_b48i = #p4hir.int<4> : !b48i
#int50_infint = #p4hir.int<50> : !infint
#int528_b16i = #p4hir.int<528> : !b16i
#int5_b32i = #p4hir.int<5> : !b32i
#int6_b32i = #p4hir.int<6> : !b32i
#int6_b48i = #p4hir.int<6> : !b48i
#int752_b16i = #p4hir.int<752> : !b16i
!headers_t = !p4hir.struct<"headers_t", ethernet: !ethernet_h>
#set_const_of_int1_b48i = #p4hir.set<const : [#int1_b48i]> : !p4hir.set<!b48i>
#set_const_of_int3_b48i = #p4hir.set<const : [#int3_b48i]> : !p4hir.set<!b48i>
#set_const_of_int4_b48i = #p4hir.set<const : [#int4_b48i]> : !p4hir.set<!b48i>
#set_mask_of_int16_b16i_int752_b16i = #p4hir.set<mask : [#int16_b16i, #int752_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int4096_b16i_int-4096_b16i = #p4hir.set<mask : [#int4096_b16i, #int-4096_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<mask : [#int4369_b16i, #int15_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int528_b16i_int752_b16i = #p4hir.set<mask : [#int528_b16i, #int752_b16i]> : !p4hir.set<!b16i>
#set_product_of_set_const_of_int1_b48i_set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<product : [#set_const_of_int1_b48i, #set_mask_of_int4369_b16i_int15_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int3_b48i_set_mask_of_int4096_b16i_int-4096_b16i = #p4hir.set<product : [#set_const_of_int3_b48i, #set_mask_of_int4096_b16i_int-4096_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int4_b48i_set_mask_of_int16_b16i_int752_b16i = #p4hir.set<product : [#set_const_of_int4_b48i, #set_mask_of_int16_b16i_int752_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int4_b48i_set_mask_of_int528_b16i_int752_b16i = #p4hir.set<product : [#set_const_of_int4_b48i, #set_mask_of_int528_b16i_int752_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
module {
  bmv2ir.header_instance @headers_t_ethernet : !p4hir.ref<!ethernet_h>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    p4hir.state @start {
      p4hir.scope {
        %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        p4corelib.extract_header %headers_t_ethernet : <!ethernet_h> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @verifyChecksum(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"})() {
    p4hir.control_apply {
    }
  }
// CHECK-LABEL:    bmv2ir.table @ingressImpl.t1
// CHECK-NEXT:     actions [@ingress::@a, @ingress::@a_params]
// CHECK-NEXT:     next_tables [#bmv2ir.action_table<@ingress::@a>, #bmv2ir.action_table<@ingress::@a_params>]
// CHECK-NEXT:     type  simple
// CHECK-NEXT:     match_type  ternary
// CHECK-NEXT:     keys [#bmv2ir.table_key<type exact, header @headers_t_ethernet["src_addr"] name = "hdr.ethernet.src_addr">, #bmv2ir.table_key<type ternary, header @headers_t_ethernet["ether_type"] name = "hdr.ethernet.ether_type">]
// CHECK-NEXT:     support_timeout false
// CHECK-NEXT:     default_entry <action @ingress::@a, action_const true, action_entry_const true>
// CHECK-NEXT:     const_entries [
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int1_b48i>, <match_type ternary first #int4369_b16i second #int15_b16i> action @ingress::@a_params action_data[#int1_b32i]>,
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int2_b48i>, <match_type ternary first #int4481_b16i second #int-1_b16i> action @ingress::@a_params action_data[#int2_b32i]>, 
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int3_b48i>, <match_type ternary first #int4096_b16i second #int-4096_b16i> action @ingress::@a_params action_data[#int3_b32i]>, 
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int4_b48i>, <match_type ternary first #int528_b16i second #int752_b16i> action @ingress::@a_params action_data[#int4_b32i]>, 
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int4_b48i>, <match_type ternary first #int16_b16i second #int752_b16i> action @ingress::@a_params action_data[#int5_b32i]>, 
// CHECK-SAME: #bmv2ir.table_entry<match_key <match_type exact first #int6_b48i>, <match_type ternary first #int0_b16i second #int0_b16i> action @ingress::@a_params action_data[#int6_b32i]>]
// CHECK-NEXT:     size 1024
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    %c6_b32i = p4hir.const #int6_b32i
    %c5_b32i = p4hir.const #int5_b32i
    %c4_b32i = p4hir.const #int4_b32i
    %c3_b32i = p4hir.const #int3_b32i
    %c2_b32i = p4hir.const #int2_b32i
    %c1_b32i = p4hir.const #int1_b32i
    %c10 = p4hir.const #int10_infint
    %false = p4hir.const #false
    p4hir.func action @a() annotations {name = "ingressImpl.a"} {
      p4hir.return
    }
    p4hir.func action @a_params(%arg3: !b32i {p4hir.annotations = {name = "param"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "param"}) annotations {name = "ingressImpl.a_params"} {
      p4hir.return
    }
    p4hir.table @ingressImpl.t1 annotations {name = "ingressImpl.t1"} {
      p4hir.table_key(%arg3: !p4hir.ref<!headers_t>) {
        %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        %src_addr_field_ref = p4hir.struct_field_ref %headers_t_ethernet["src_addr"] : <!ethernet_h>
        %val = p4hir.read %src_addr_field_ref : <!b48i>
        p4hir.match_key #exact %val : !b48i annotations {name = "hdr.ethernet.src_addr"}
        %headers_t_ethernet_0 = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        %ether_type_field_ref = p4hir.struct_field_ref %headers_t_ethernet_0["ether_type"] : <!ethernet_h>
        %val_1 = p4hir.read %ether_type_field_ref : <!b16i>
        p4hir.match_key #ternary %val_1 : !b16i annotations {name = "hdr.ethernet.ether_type"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_params(%arg3: !b32i {p4hir.annotations = {}, p4hir.param_name = "param"}) {
          p4hir.call @ingress::@a_params (%arg3) : (!b32i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      %largest_priority_wins = p4hir.table_property "largest_priority_wins" {
        p4hir.yield %false : !p4hir.bool
      } : !p4hir.bool
      %priority_delta = p4hir.table_property "priority_delta" {
        p4hir.yield %c10 : !infint
      } : !infint
      p4hir.table_entries {
        p4hir.table_entry const #set_product_of_set_const_of_int1_b48i_set_mask_of_int4369_b16i_int15_b16i priority = #int10_infint {
          p4hir.call @ingress::@a_params (%c1_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int2_b48i, #int4481_b16i]> : tuple<!b48i, !b16i> priority = #int20_infint {
          p4hir.call @ingress::@a_params (%c2_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #set_product_of_set_const_of_int3_b48i_set_mask_of_int4096_b16i_int-4096_b16i priority = #int30_infint {
          p4hir.call @ingress::@a_params (%c3_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry const #set_product_of_set_const_of_int4_b48i_set_mask_of_int528_b16i_int752_b16i priority = #int40_infint {
          p4hir.call @ingress::@a_params (%c4_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #set_product_of_set_const_of_int4_b48i_set_mask_of_int16_b16i_int752_b16i priority = #int40_infint {
          p4hir.call @ingress::@a_params (%c5_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int6_b48i, #everything]> : tuple<!b48i, !p4hir.set<!p4hir.dontcare>> priority = #int50_infint {
          p4hir.call @ingress::@a_params (%c6_b32i) : (!b32i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %ingressImpl.t1_apply_result = p4hir.table_apply @ingress::@ingressImpl.t1 with key(%arg0) : (!p4hir.ref<!headers_t>) -> !t1_0
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @updateChecksum(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
      %val = p4hir.read %headers_t_ethernet : <!ethernet_h>
      p4corelib.emit %val : !ethernet_h to %arg0 : !p4corelib.packet_out
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @verifyChecksum, ingress @ingress, egress @egress, compute_checksum @updateChecksum, deparser @deparser
}

// -----

!Meta_t = !p4hir.struct<"Meta_t">
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!packet_in = !p4hir.extern<"packet_in">
!packet_out = !p4hir.extern<"packet_out">
!string = !p4hir.string
!type_D = !p4hir.type_var<"D">
!type_H = !p4hir.type_var<"H">
!type_M = !p4hir.type_var<"M">
!type_O = !p4hir.type_var<"O">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#in = #p4hir<dir in>
#inout = #p4hir<dir inout>
#lpm = #p4hir.match_kind<"lpm">
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!hdr = !p4hir.header<"hdr", e: !b8i, t: !b16i, l: !b8i, r: !b8i, v: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t_lpm_0 = !p4hir.struct<"t_lpm_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !b32i>
#int-16_b8i = #p4hir.int<240> : !b8i
#int0_b9i = #p4hir.int<0> : !b9i
#int11_b9i = #p4hir.int<11> : !b9i
#int12_b9i = #p4hir.int<12> : !b9i
#int13_b9i = #p4hir.int<13> : !b9i
#int17_b8i = #p4hir.int<17> : !b8i
#int18_b8i = #p4hir.int<18> : !b8i
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!Header_t = !p4hir.struct<"Header_t", h: !hdr>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
#set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<mask : [#int17_b8i, #int-16_b8i]> : !p4hir.set<!b8i>
#set_product_of_set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<product : [#set_mask_of_int17_b8i_int-16_b8i]> : !p4hir.set<tuple<!b8i>>
module {
  bmv2ir.header_instance @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
  bmv2ir.header_instance @Header_t_h : !p4hir.ref<!hdr>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #undir, p4hir.param_name = "b"}, %arg1: !p4hir.ref<!Header_t> {p4hir.dir = #out, p4hir.param_name = "h"}, %arg2: !p4hir.ref<!Meta_t> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg3: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.state @start {
      p4hir.scope {
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        p4corelib.extract_header %Header_t_h : <!hdr> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @vrfy(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #inout, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @update(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #inout, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #undir, p4hir.param_name = "b"}, %arg1: !Header_t {p4hir.dir = #in, p4hir.param_name = "h"})() {
    p4hir.control_apply {
      %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
      %val = p4hir.read %Header_t_h : <!hdr>
      p4corelib.emit %val : !hdr to %arg0 : !p4corelib.packet_out
    }
  }
  p4hir.control @ingress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "standard_meta"})() {
    %c13_b9i = p4hir.const #int13_b9i
    %c12_b9i = p4hir.const #int12_b9i
    %c11_b9i = p4hir.const #int11_b9i
    p4hir.func action @a() annotations {name = "ingress.a"} {
      %c0_b9i = p4hir.const #int0_b9i
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %c0_b9i, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.func action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {name = "x"}, p4hir.dir = #undir, p4hir.param_name = "x"}) annotations {name = "ingress.a_with_control_params"} {
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %arg3, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }


// CHECK:    bmv2ir.table @ingress.t_lpm
// CHECK:     actions [@ingress::@a, @ingress::@a_with_control_params]
// CHECK:     next_tables [#bmv2ir.action_table<@ingress::@a>, #bmv2ir.action_table<@ingress::@a_with_control_params>]
// CHECK:     type  simple
// CHECK:     match_type  lpm
// CHECK:     keys [#bmv2ir.table_key<type lpm, header @Header_t_h["l"] name = "h.h.l">]
// CHECK:     support_timeout false
// CHECK:     default_entry <action @ingress::@a, action_const true, action_entry_const true>
// CHECK:     const_entries [#bmv2ir.table_entry<match_key <match_type lpm first #int17_b8i second 4 : si64> action @ingress::@a_with_control_params action_data[#int11_b9i]>, #bmv2ir.table_entry<match_key <match_type lpm first #int18_b8i second 8 : si64> action @ingress::@a_with_control_params action_data[#int12_b9i]>, #bmv2ir.table_entry<match_key <match_type lpm first #int0_b8i second 0 : si64> action @ingress::@a_with_control_params action_data[#int13_b9i]>]
// CHECK:     size 1024
// CHECK:  }

    p4hir.table @ingress.t_lpm annotations {name = "ingress.t_lpm"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Header_t>) {
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        %l_field_ref = p4hir.struct_field_ref %Header_t_h["l"] : <!hdr>
        %val = p4hir.read %l_field_ref : <!b8i>
        p4hir.match_key #lpm %val : !b8i annotations {name = "h.h.l"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {}, p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg3) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #set_product_of_set_mask_of_int17_b8i_int-16_b8i {
          p4hir.call @ingress::@a_with_control_params (%c11_b9i) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int18_b8i]> : tuple<!b8i> {
          p4hir.call @ingress::@a_with_control_params (%c12_b9i) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#everything]> : tuple<!p4hir.set<!p4hir.dontcare>> {
          p4hir.call @ingress::@a_with_control_params (%c13_b9i) : (!b9i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %ingress.t_lpm_apply_result = p4hir.table_apply @ingress::@ingress.t_lpm with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_lpm_0
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @vrfy, ingress @ingress, egress @egress, compute_checksum @update, deparser @deparser
}

// -----

!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!metadata_t = !p4hir.struct<"metadata_t">
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#exact = #p4hir.match_kind<"exact">
#false = #p4hir.bool<false> : !p4hir.bool
#ternary = #p4hir.match_kind<"ternary">
!ethernet_h = !p4hir.header<"ethernet_h", dst_addr: !b48i, src_addr: !b48i, ether_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t1_0 = !p4hir.struct<"t1_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !b32i>
#int-4096_b16i = #p4hir.int<61440> : !b16i
#int10_infint = #p4hir.int<10> : !infint
#int15_b16i = #p4hir.int<15> : !b16i
#int16_b16i = #p4hir.int<16> : !b16i
#int1_b32i = #p4hir.int<1> : !b32i
#int1_b48i = #p4hir.int<1> : !b48i
#int20_infint = #p4hir.int<20> : !infint
#int2_b32i = #p4hir.int<2> : !b32i
#int2_b48i = #p4hir.int<2> : !b48i
#int30_infint = #p4hir.int<30> : !infint
#int3_b32i = #p4hir.int<3> : !b32i
#int3_b48i = #p4hir.int<3> : !b48i
#int4096_b16i = #p4hir.int<4096> : !b16i
#int40_infint = #p4hir.int<40> : !infint
#int4369_b16i = #p4hir.int<4369> : !b16i
#int4481_b16i = #p4hir.int<4481> : !b16i
#int4_b32i = #p4hir.int<4> : !b32i
#int4_b48i = #p4hir.int<4> : !b48i
#int50_infint = #p4hir.int<50> : !infint
#int528_b16i = #p4hir.int<528> : !b16i
#int5_b32i = #p4hir.int<5> : !b32i
#int6_b32i = #p4hir.int<6> : !b32i
#int6_b48i = #p4hir.int<6> : !b48i
#int752_b16i = #p4hir.int<752> : !b16i
!headers_t = !p4hir.struct<"headers_t", ethernet: !ethernet_h>
#set_const_of_int1_b48i = #p4hir.set<const : [#int1_b48i]> : !p4hir.set<!b48i>
#set_const_of_int3_b48i = #p4hir.set<const : [#int3_b48i]> : !p4hir.set<!b48i>
#set_const_of_int4_b48i = #p4hir.set<const : [#int4_b48i]> : !p4hir.set<!b48i>
#set_mask_of_int16_b16i_int752_b16i = #p4hir.set<mask : [#int16_b16i, #int752_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int4096_b16i_int-4096_b16i = #p4hir.set<mask : [#int4096_b16i, #int-4096_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<mask : [#int4369_b16i, #int15_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int528_b16i_int752_b16i = #p4hir.set<mask : [#int528_b16i, #int752_b16i]> : !p4hir.set<!b16i>
#set_product_of_set_const_of_int1_b48i_set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<product : [#set_const_of_int1_b48i, #set_mask_of_int4369_b16i_int15_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int3_b48i_set_mask_of_int4096_b16i_int-4096_b16i = #p4hir.set<product : [#set_const_of_int3_b48i, #set_mask_of_int4096_b16i_int-4096_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int4_b48i_set_mask_of_int16_b16i_int752_b16i = #p4hir.set<product : [#set_const_of_int4_b48i, #set_mask_of_int16_b16i_int752_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
#set_product_of_set_const_of_int4_b48i_set_mask_of_int528_b16i_int752_b16i = #p4hir.set<product : [#set_const_of_int4_b48i, #set_mask_of_int528_b16i_int752_b16i]> : !p4hir.set<tuple<!b48i, !b16i>>
module {
  bmv2ir.header_instance @headers_t_ethernet : !p4hir.ref<!ethernet_h>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    p4hir.state @start {
      p4hir.scope {
        %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        p4corelib.extract_header %headers_t_ethernet : <!ethernet_h> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @verifyChecksum(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    %c6_b32i = p4hir.const #int6_b32i
    %c5_b32i = p4hir.const #int5_b32i
    %c4_b32i = p4hir.const #int4_b32i
    %c3_b32i = p4hir.const #int3_b32i
    %c2_b32i = p4hir.const #int2_b32i
    %c1_b32i = p4hir.const #int1_b32i
    %c10 = p4hir.const #int10_infint
    %false = p4hir.const #false
    p4hir.func action @a() annotations {name = "ingressImpl.a"} {
      p4hir.return
    }
    p4hir.func action @a_params(%arg3: !b32i {p4hir.annotations = {name = "param"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "param"}) annotations {name = "ingressImpl.a_params"} {
      p4hir.return
    }
    p4hir.table @ingressImpl.t1 annotations {name = "ingressImpl.t1"} {
      p4hir.table_key(%arg3: !p4hir.ref<!headers_t>) {
        %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        %src_addr_field_ref = p4hir.struct_field_ref %headers_t_ethernet["src_addr"] : <!ethernet_h>
        %val = p4hir.read %src_addr_field_ref : <!b48i>
        p4hir.match_key #exact %val : !b48i annotations {name = "hdr.ethernet.src_addr"}
        %headers_t_ethernet_0 = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
        %ether_type_field_ref = p4hir.struct_field_ref %headers_t_ethernet_0["ether_type"] : <!ethernet_h>
        %val_1 = p4hir.read %ether_type_field_ref : <!b16i>
        p4hir.match_key #ternary %val_1 : !b16i annotations {name = "hdr.ethernet.ether_type"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_params(%arg3: !b32i {p4hir.annotations = {}, p4hir.param_name = "param"}) {
          p4hir.call @ingress::@a_params (%arg3) : (!b32i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      %largest_priority_wins = p4hir.table_property "largest_priority_wins" {
        p4hir.yield %false : !p4hir.bool
      } : !p4hir.bool
      %priority_delta = p4hir.table_property "priority_delta" {
        p4hir.yield %c10 : !infint
      } : !infint
      p4hir.table_entries {
        p4hir.table_entry const #set_product_of_set_const_of_int1_b48i_set_mask_of_int4369_b16i_int15_b16i priority = #int10_infint {
          p4hir.call @ingress::@a_params (%c1_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int2_b48i, #int4481_b16i]> : tuple<!b48i, !b16i> priority = #int20_infint {
          p4hir.call @ingress::@a_params (%c2_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #set_product_of_set_const_of_int3_b48i_set_mask_of_int4096_b16i_int-4096_b16i priority = #int30_infint {
          p4hir.call @ingress::@a_params (%c3_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry const #set_product_of_set_const_of_int4_b48i_set_mask_of_int528_b16i_int752_b16i priority = #int40_infint {
          p4hir.call @ingress::@a_params (%c4_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #set_product_of_set_const_of_int4_b48i_set_mask_of_int16_b16i_int752_b16i priority = #int40_infint {
          p4hir.call @ingress::@a_params (%c5_b32i) : (!b32i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int6_b48i, #everything]> : tuple<!b48i, !p4hir.set<!p4hir.dontcare>> priority = #int50_infint {
          p4hir.call @ingress::@a_params (%c6_b32i) : (!b32i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %ingressImpl.t1_apply_result = p4hir.table_apply @ingress::@ingressImpl.t1 with key(%arg0) : (!p4hir.ref<!headers_t>) -> !t1_0
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @updateChecksum(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "umd"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      %headers_t_ethernet = bmv2ir.symbol_ref @headers_t_ethernet : !p4hir.ref<!ethernet_h>
      %val = p4hir.read %headers_t_ethernet : <!ethernet_h>
      p4corelib.emit %val : !ethernet_h to %arg0 : !p4corelib.packet_out
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @verifyChecksum, ingress @ingress, egress @egress, compute_checksum @updateChecksum, deparser @deparser
}

// -----
!Meta_t = !p4hir.struct<"Meta_t">
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#lpm = #p4hir.match_kind<"lpm">
!hdr = !p4hir.header<"hdr", e: !b8i, t: !b16i, l: !b8i, r: !b8i, v: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t_lpm_0 = !p4hir.struct<"t_lpm_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !b32i>
#int-16_b8i = #p4hir.int<240> : !b8i
#int0_b9i = #p4hir.int<0> : !b9i
#int11_b9i = #p4hir.int<11> : !b9i
#int12_b9i = #p4hir.int<12> : !b9i
#int13_b9i = #p4hir.int<13> : !b9i
#int17_b8i = #p4hir.int<17> : !b8i
#int18_b8i = #p4hir.int<18> : !b8i
!Header_t = !p4hir.struct<"Header_t", h: !hdr>
#set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<mask : [#int17_b8i, #int-16_b8i]> : !p4hir.set<!b8i>
#set_product_of_set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<product : [#set_mask_of_int17_b8i_int-16_b8i]> : !p4hir.set<tuple<!b8i>>
module {
  bmv2ir.header_instance @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
  bmv2ir.header_instance @Header_t_h : !p4hir.ref<!hdr>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}, %arg1: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "h"}, %arg2: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg3: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "sm"})() {
    p4hir.state @start {
      p4hir.scope {
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        p4corelib.extract_header %Header_t_h : <!hdr> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @vrfy(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @update(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "sm"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}, %arg1: !Header_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h"})() {
    p4hir.control_apply {
      %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
      %val = p4hir.read %Header_t_h : <!hdr>
      p4corelib.emit %val : !hdr to %arg0 : !p4corelib.packet_out
    }
  }
  p4hir.control @ingress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_meta"})() {
    %c13_b9i = p4hir.const #int13_b9i
    %c12_b9i = p4hir.const #int12_b9i
    %c11_b9i = p4hir.const #int11_b9i
    p4hir.func action @a() annotations {name = "ingress.a"} {
      %c0_b9i = p4hir.const #int0_b9i
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %c0_b9i, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.func action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {name = "x"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}) annotations {name = "ingress.a_with_control_params"} {
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %arg3, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.table @ingress.t_lpm annotations {name = "ingress.t_lpm"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Header_t>) {
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        %l_field_ref = p4hir.struct_field_ref %Header_t_h["l"] : <!hdr>
        %val = p4hir.read %l_field_ref : <!b8i>
        p4hir.match_key #lpm %val : !b8i annotations {name = "h.h.l"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {}, p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg3) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #set_product_of_set_mask_of_int17_b8i_int-16_b8i {
          p4hir.call @ingress::@a_with_control_params (%c11_b9i) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int18_b8i]> : tuple<!b8i> {
          p4hir.call @ingress::@a_with_control_params (%c12_b9i) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#everything]> : tuple<!p4hir.set<!p4hir.dontcare>> {
          p4hir.call @ingress::@a_with_control_params (%c13_b9i) : (!b9i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %ingress.t_lpm_apply_result = p4hir.table_apply @ingress::@ingress.t_lpm with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_lpm_0
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @vrfy, ingress @ingress, egress @egress, compute_checksum @update, deparser @deparser
}

// -----
!Meta_t = !p4hir.struct<"Meta_t">
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!validity_bit = !p4hir.validity.bit
#exact = #p4hir.match_kind<"exact">
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
!hdr = !p4hir.header<"hdr", e: !b8i, t: !b16i, l: !b8i, r: !b8i, v: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t_valid_0 = !p4hir.struct<"t_valid_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !b32i>
#int0_b9i = #p4hir.int<0> : !b9i
#int1_b8i = #p4hir.int<1> : !b8i
#int1_b9i = #p4hir.int<1> : !b9i
#int2_b8i = #p4hir.int<2> : !b8i
#int2_b9i = #p4hir.int<2> : !b9i
#valid = #p4hir<validity.bit valid> : !validity_bit
!Header_t = !p4hir.struct<"Header_t", h: !hdr>
module {
  bmv2ir.header_instance @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
  bmv2ir.header_instance @Header_t_h : !p4hir.ref<!hdr>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}, %arg1: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "h"}, %arg2: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg3: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "sm"})() {
    p4hir.state @start {
      p4hir.scope {
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        p4corelib.extract_header %Header_t_h : <!hdr> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @vrfy(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @update(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "sm"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b"}, %arg1: !Header_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h"})() {
    p4hir.control_apply {
      %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
      %val = p4hir.read %Header_t_h : <!hdr>
      p4corelib.emit %val : !hdr to %arg0 : !p4corelib.packet_out
    }
  }
  p4hir.control @ingress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_meta"})() {
    %c2_b9i = p4hir.const #int2_b9i
    %c1_b9i = p4hir.const #int1_b9i
    p4hir.func action @a() annotations {name = "ingress.a"} {
      %c0_b9i = p4hir.const #int0_b9i
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %c0_b9i, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.func action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {name = "x"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}) annotations {name = "ingress.a_with_control_params"} {
      %standard_metadata_t = bmv2ir.symbol_ref @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %standard_metadata_t["egress_spec"] : <!standard_metadata_t>
      p4hir.assign %arg3, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
// CHECK:    bmv2ir.table @ingress.t_valid
// CHECK:     actions [@ingress::@a, @ingress::@a_with_control_params]
// CHECK:     next_tables [#bmv2ir.action_table<@ingress::@a>, #bmv2ir.action_table<@ingress::@a_with_control_params>]
// CHECK:     type  simple
// CHECK:     match_type  exact
// CHECK:     keys [#bmv2ir.table_key<type exact, header @Header_t_h["$valid$"] name = "h.h.$valid$">, #bmv2ir.table_key<type exact, header @Header_t_h["e"] name = "h.h.e">]
// CHECK:     support_timeout false
// CHECK:     default_entry <action @ingress::@a, action_const true, action_entry_const true>
// CHECK:     const_entries [#bmv2ir.table_entry<match_key <match_type exact first #int-1_b1i>, <match_type exact first #int1_b8i> action @ingress::@a_with_control_params action_data[#int1_b9i]>, #bmv2ir.table_entry<match_key <match_type exact first #int0_b1i>, <match_type exact first #int2_b8i> action @ingress::@a_with_control_params action_data[#int2_b9i]>]
// CHECK:     size 1024
    p4hir.table @ingress.t_valid annotations {name = "ingress.t_valid"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Header_t>) {
        %valid = p4hir.const #valid
        %Header_t_h = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        %__valid_field_ref = p4hir.struct_field_ref %Header_t_h["__valid"] : <!hdr>
        %val = p4hir.read %__valid_field_ref : <!validity_bit>
        %eq = p4hir.cmp(eq, %val : !validity_bit, %valid : !validity_bit)
        p4hir.match_key #exact %eq : !p4hir.bool annotations {name = "h.h.$valid$"}
        %Header_t_h_0 = bmv2ir.symbol_ref @Header_t_h : !p4hir.ref<!hdr>
        %e_field_ref = p4hir.struct_field_ref %Header_t_h_0["e"] : <!hdr>
        %val_1 = p4hir.read %e_field_ref : <!b8i>
        p4hir.match_key #exact %val_1 : !b8i annotations {name = "h.h.e"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg3: !b9i {p4hir.annotations = {}, p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg3) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #p4hir.aggregate<[#true, #int1_b8i]> : tuple<!p4hir.bool, !b8i> {
          p4hir.call @ingress::@a_with_control_params (%c1_b9i) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#false, #int2_b8i]> : tuple<!p4hir.bool, !b8i> {
          p4hir.call @ingress::@a_with_control_params (%c2_b9i) : (!b9i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %ingress.t_valid_apply_result = p4hir.table_apply @ingress::@ingress.t_valid with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_valid_0
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @vrfy, ingress @ingress, egress @egress, compute_checksum @update, deparser @deparser
}
