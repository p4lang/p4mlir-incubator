// RUN: p4mlir-opt -p='builtin.module(lower-to-header-instance)' --split-input-file %s | FileCheck %s
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
!header_bottom = !p4hir.header<"header_bottom", length: !b8i, data: !p4hir.varbit<256>, __valid: !validity_bit>
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
#int1_b8i = #p4hir.int<1> : !b8i
#int256_b32i = #p4hir.int<256> : !b32i
#int2_b8i = #p4hir.int<2> : !b8i
!Headers_t = !p4hir.struct<"Headers_t", top: !header_top, one: !header_one, two: !header_two, bottom: !header_bottom>
!header_and_bit = !p4hir.struct<"header_and_bit", top: !header_top, bit: !b8i>
module {
// CHECK:  bmv2ir.header_instance @e_0_var_0 : !p4hir.ref<!header_one>
// CHECK:  bmv2ir.header_instance @Headers_t_top : !p4hir.ref<!header_top>
// CHECK:  bmv2ir.header_instance @Headers_t_one : !p4hir.ref<!header_one>
// CHECK:  bmv2ir.header_instance @Headers_t_two : !p4hir.ref<!header_two>
  p4hir.parser @prs(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %e_0 = p4hir.variable ["e_0"] annotations {name = "ParserImpl.e"} : <!header_one>
    // CHECK: %[[E_0:.*]] = bmv2ir.symbol_ref @e_0_var_0 : !p4hir.ref<!header_one>
    p4hir.state @start {
      %top_field_ref = p4hir.struct_field_ref %arg1["top"] : <!Headers_t>
      p4corelib.extract_header %top_field_ref : <!header_top> from %arg0 : !p4corelib.packet_in
// CHECK:  %[[REF:.*]] = bmv2ir.symbol_ref @Headers_t_top : !p4hir.ref<!header_top>
// CHECK: p4corelib.extract_header %[[REF]] : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_headers
    }
    p4hir.state @parse_headers {
      %lookahead = p4corelib.packet_lookahead %arg0 : !p4corelib.packet_in -> !b8i
      p4hir.transition_select %lookahead : !b8i {
        p4hir.select_case {
          %c1_b8i = p4hir.const #int1_b8i
          %set = p4hir.set (%c1_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @prs::@parse_one
        p4hir.select_case {
          %c2_b8i = p4hir.const #int2_b8i
          %set = p4hir.set (%c2_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @prs::@parse_two
        p4hir.select_case {
          %c1_b8i = p4hir.const #int1_b8i
          %c2_b8i = p4hir.const #int2_b8i
          %mask = p4hir.mask(%c1_b8i, %c2_b8i) : !p4hir.set<!b8i>
          p4hir.yield %mask : !p4hir.set<!b8i>
        } to @prs::@parse_two
        p4hir.select_case {
          %everything = p4hir.const #everything
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @prs::@parse_bottom
      }
    }
    p4hir.state @parse_one {
// CHECK: %[[REF2:.*]] = bmv2ir.symbol_ref @Headers_t_one : !p4hir.ref<!header_one>
      %one_field_ref = p4hir.struct_field_ref %arg1["one"] : <!Headers_t>
      p4corelib.extract_header %e_0 : <!header_one> from %arg0 : !p4corelib.packet_in
// CHECK: p4corelib.extract_header %[[E_0]] : <!header_one> from %arg0 : !p4corelib.packet_in
      %val = p4hir.read %e_0 : <!header_one>
      p4hir.assign %val, %one_field_ref : <!header_one>
// CHECK: p4hir.assign %{{.*}}, %[[REF2]] : <!header_one>
      p4hir.transition to @prs::@parse_two
    }
    p4hir.state @parse_two {
// CHECK: %[[REF3:.*]] = bmv2ir.symbol_ref @Headers_t_two : !p4hir.ref<!header_two>
      %two_field_ref = p4hir.struct_field_ref %arg1["two"] : <!Headers_t>
      p4corelib.extract_header %two_field_ref : <!header_two> from %arg0 : !p4corelib.packet_in
// CHECK:  p4corelib.extract_header %[[REF3]] : <!header_two> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_bottom
    }
    p4hir.state @parse_bottom {
      p4hir.transition to @prs::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs::@start
  }
  // Check that the Headers_t arg now leads to the same instance being used for its fields
  p4hir.parser @other(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    p4hir.state @start {
      %top_field_ref = p4hir.struct_field_ref %arg1["top"] : <!Headers_t>
      p4corelib.extract_header %top_field_ref : <!header_top> from %arg0 : !p4corelib.packet_in
// CHECK:  %[[REF:.*]] = bmv2ir.symbol_ref @Headers_t_top : !p4hir.ref<!header_top>
// CHECK: p4corelib.extract_header %[[REF]] : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs::@start
  }
}

// -----

// Checks that we correctly handle a header parser arg
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
module {
// CHECK: bmv2ir.header_instance @header_top : !p4hir.ref<!header_top>
  p4hir.parser @prs_header_arg(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    p4hir.state @start {
// CHECK:      %[[REF:.*]] = bmv2ir.symbol_ref @header_top : !p4hir.ref<!header_top>
// CHECK:      p4corelib.extract_header %[[REF]] : <!header_top> from %arg0 : !p4corelib.packet_in
      p4corelib.extract_header %arg1 : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs_header_arg::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_header_arg::@start
  }

}

// -----

// Checks that we correctly split a struct having both headers and bits
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_and_bit = !p4hir.struct<"header_and_bit", top: !header_top, bit: !b8i>
// CHECK: ![[SPLIT_STRUCT:.*]] = !p4hir.struct<"header_and_bit", bit: !b8i>
module {
// CHECK: bmv2ir.header_instance @header_and_bit : !p4hir.ref<![[SPLIT_STRUCT]]>
    p4hir.parser @prs_header_and_bit(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!header_and_bit> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %var = p4hir.variable ["top_0"] annotations {name = "ParserImpl.e"} : <!header_top>
    p4hir.state @start {
      p4corelib.extract_header %var : <!header_top> from %arg0 : !p4corelib.packet_in
      %bit = p4hir.struct_field_ref %var["skip"] : <!header_top>
      %val = p4hir.read %bit : <!b8i>
      %ref = p4hir.struct_field_ref %arg1["bit"] : <!header_and_bit>
// CHECK: %[[REF:.*]] = bmv2ir.symbol_ref @header_and_bit : !p4hir.ref<!header_and_bit>
// CHECK: %{{.*}} = p4hir.struct_field_ref %[[REF]]["bit"] : <!header_and_bit>
      p4hir.assign %val, %ref : <!b8i>
      p4hir.transition to @prs_header_and_bit::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_header_and_bit::@start
  }
}

// -----

// Checks that we correctly insert header instances for structs with only bit fields
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!bit_only = !p4hir.struct<"bit_only", bit: !b8i>
module {
// CHECK: bmv2ir.header_instance @bit_only : !p4hir.ref<!bit_only>
    p4hir.parser @prs_only_bit(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!bit_only> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"}, %arg2: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %var = p4hir.variable ["top_0"] annotations {name = "ParserImpl.e"} : <!header_top>
    p4hir.state @start {
      p4corelib.extract_header %var : <!header_top> from %arg0 : !p4corelib.packet_in
      %bit = p4hir.struct_field_ref %var["skip"] : <!header_top>
      %val = p4hir.read %bit : <!b8i>
      %ref = p4hir.struct_field_ref %arg1["bit"] : <!bit_only>
      p4hir.assign %val, %ref : <!b8i>
      %val2 = p4hir.read %ref : <!b8i>
      %ref2 = p4hir.struct_field_ref %arg2["skip"] : <!header_top>
      p4hir.assign %val2, %ref2 : <!b8i>
      p4hir.transition to @prs_only_bit::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_only_bit::@start
  }
}

// -----

!infint = !p4hir.infint
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!b1i = !p4hir.bit<1>
!b3i = !p4hir.bit<3>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!b12i = !p4hir.bit<12>
!b13i = !p4hir.bit<13>
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b32i = !p4hir.bit<32>
!b48i = !p4hir.bit<48>
!validity_bit = !p4hir.validity.bit
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
!ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i>
// CHECK: !ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i, _padding: !b4i>
!anon = !p4hir.enum<on_miss, rewrite_src_dst_mac, NoAction_2>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!rewrite_mac_0 = !p4hir.struct<"rewrite_mac_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#inout = #p4hir<dir inout>
#undir = #p4hir<dir undir>
#exact = #p4hir.match_kind<"exact">
#int32768_infint = #p4hir.int<32768> : !infint
#int-1_b8i = #p4hir.int<255> : !b8i
module {
// CHECK: bmv2ir.header_instance @ingress_metadata_t : !p4hir.ref<!ingress_metadata_t>
// CHECK: bmv2ir.header_instance @headers_ethernet : !p4hir.ref<!ethernet_t>
// CHECK: bmv2ir.header_instance @headers_ipv4 : !p4hir.ref<!ipv4_t>
  p4hir.control @egress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!ingress_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "standard_metadata"})() {
    p4hir.control_local @__local_egress_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_egress_meta_0 = %arg1 : !p4hir.ref<!ingress_metadata_t>
    p4hir.control_local @__local_egress_standard_metadata_0 = %arg2 : !p4hir.ref<!standard_metadata_t>
// CHECK-NOT: p4hir.control_local
    p4hir.func action @NoAction_2() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @on_miss() annotations {name = "egress.on_miss"} {
      p4hir.return
    }
// CHECK-LABEL: p4hir.func action @rewrite_src_dst_mac
    p4hir.func action @rewrite_src_dst_mac(%arg3: !b48i {p4hir.annotations = {name = "smac"}, p4hir.dir = #undir, p4hir.param_name = "smac"}, %arg4: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.dir = #undir, p4hir.param_name = "dmac"}) annotations {name = "egress.rewrite_src_dst_mac"} {
      %__local_egress_hdr_0 = p4hir.symbol_ref @egress::@__local_egress_hdr_0 : !p4hir.ref<!headers>
// CHECK-NOT: p4hir.symbol_ref
// CHECK: %[[REF:.*]] = bmv2ir.symbol_ref @headers_ethernet : !p4hir.ref<!ethernet_t>
// CHECK: %{{.*}} = p4hir.struct_field_ref %[[REF]]["srcAddr"] : <!ethernet_t>
      %ethernet_field_ref = p4hir.struct_field_ref %__local_egress_hdr_0["ethernet"] : <!headers>
      %srcAddr_field_ref = p4hir.struct_field_ref %ethernet_field_ref["srcAddr"] : <!ethernet_t>
      p4hir.assign %arg3, %srcAddr_field_ref : <!b48i>
      %__local_egress_hdr_0_0 = p4hir.symbol_ref @egress::@__local_egress_hdr_0 : !p4hir.ref<!headers>
      %ethernet_field_ref_1 = p4hir.struct_field_ref %__local_egress_hdr_0_0["ethernet"] : <!headers>
      %dstAddr_field_ref = p4hir.struct_field_ref %ethernet_field_ref_1["dstAddr"] : <!ethernet_t>
      p4hir.assign %arg4, %dstAddr_field_ref : <!b48i>
      %__local_egress_meta_0 = p4hir.symbol_ref @egress::@__local_egress_meta_0 : !p4hir.ref<!ingress_metadata_t>
      %vrf_ref = p4hir.struct_field_ref %__local_egress_meta_0["vrf"] : <!ingress_metadata_t>
// CHECK: %[[REF2:.*]] = bmv2ir.symbol_ref @ingress_metadata_t : !p4hir.ref<!ingress_metadata_t>
// CHECK: %{{.*}} = p4hir.struct_field_ref %[[REF2]]["vrf"] : <!ingress_metadata_t>
      p4hir.return
    }
// CHECK-LABEL: p4hir.func action @fib_hit_nexthop
    p4hir.func action @fib_hit_nexthop(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #undir, p4hir.param_name = "nexthop_index_1"}) annotations {name = "ingress.fib_hit_nexthop"} {
      %__local_ingress_meta_0 = p4hir.symbol_ref @egress::@__local_egress_meta_0 : !p4hir.ref<!ingress_metadata_t>
      %vrf_field_ref = p4hir.struct_field_ref %__local_ingress_meta_0["nexthop_index"] : <!ingress_metadata_t>
      p4hir.assign %arg3, %vrf_field_ref : <!b16i>
      %__local_ingress_hdr_0 = p4hir.symbol_ref @egress::@__local_egress_hdr_0 : !p4hir.ref<!headers>
      %ipv4_field_ref = p4hir.struct_field_ref %__local_ingress_hdr_0["ipv4"] : <!headers>
      %ttl_field_ref = p4hir.struct_field_ref %ipv4_field_ref["ttl"] : <!ipv4_t>
// CHECK:       %[[REF_V4:.*]] = bmv2ir.symbol_ref @headers_ipv4 : !p4hir.ref<!ipv4_t>
// CHECK-NEXT:  %{{.*}} = p4hir.struct_field_ref %[[REF_V4]]["ttl"] : <!ipv4_t>
      %__local_ingress_hdr_0_0 = p4hir.symbol_ref @egress::@__local_egress_hdr_0 : !p4hir.ref<!headers>
      %val = p4hir.read %__local_ingress_hdr_0_0 : <!headers>
      %ipv4 = p4hir.struct_extract %val["ipv4"] : !headers
      %ttl = p4hir.struct_extract %ipv4["ttl"] : !ipv4_t
// CHECK:      %[[REF_V42:.*]] = bmv2ir.symbol_ref @headers_ipv4 : !p4hir.ref<!ipv4_t>
// CHECK:      %[[FREF_V42:.*]] = p4hir.read %[[REF_V42]] : <!ipv4_t>
// CHECK:      %{{.*}} = p4hir.struct_extract %[[FREF_V42]]["ttl"] : !ipv4_t
      %c-1_b8i = p4hir.const #int-1_b8i
      %add = p4hir.binop(add, %ttl, %c-1_b8i) : !b8i
      p4hir.assign %add, %ttl_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.table @rewrite_mac_0 annotations {name = "egress.rewrite_mac"} {
      p4hir.table_actions {
        p4hir.table_action @on_miss() {
          p4hir.call @egress::@on_miss () : () -> ()
        }
        p4hir.table_action @rewrite_src_dst_mac(%arg3: !b48i {p4hir.annotations = {name = "smac"}, p4hir.param_name = "smac"}, %arg4: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.param_name = "dmac"}) {
          p4hir.call @egress::@rewrite_src_dst_mac (%arg3, %arg4) : (!b48i, !b48i) -> ()
        }
        p4hir.table_action @NoAction_2() annotations {defaultonly} {
          p4hir.call @egress::@NoAction_2 () : () -> ()
        }
      }
      p4hir.table_key(%arg3 : !p4hir.ref<!ingress_metadata_t>) {
        %nexthop_index_ref = p4hir.struct_field_ref %arg3["nexthop_index"] : <!ingress_metadata_t>
        %nexthop_index = p4hir.read %nexthop_index_ref : <!b16i>
        p4hir.match_key #exact %nexthop_index : !b16i annotations {name = "meta.ingress_metadata.nexthop_index"}
      }
      %size = p4hir.table_size #int32768_infint
      p4hir.table_default_action {
        p4hir.call @egress::@NoAction_2 () : () -> ()
      }
    }
// CHECK-LABEL: p4hir.table @ipv4_fib_0
    p4hir.table @ipv4_fib_0 annotations {name = "ingress.ipv4_fib"} {
      p4hir.table_actions {
        p4hir.table_action @on_miss() {
          p4hir.call @egress::@on_miss () : () -> ()
        }
        p4hir.table_action @rewrite_src_dst_mac(%arg3: !b48i {p4hir.annotations = {name = "smac"}, p4hir.param_name = "smac"}, %arg4: !b48i {p4hir.annotations = {name = "dmac"}, p4hir.param_name = "dmac"}) {
          p4hir.call @egress::@rewrite_src_dst_mac (%arg3, %arg4) : (!b48i, !b48i) -> ()
        }
        p4hir.table_action @NoAction_2() annotations {defaultonly} {
          p4hir.call @egress::@NoAction_2 () : () -> ()
        }
      }
      p4hir.table_key(%arg3 : !p4hir.ref<!headers>, %arg4 : !p4hir.ref<!ingress_metadata_t>) {
        %val = p4hir.read %arg4 : <!ingress_metadata_t>
        %vrf = p4hir.struct_extract %val["vrf"] : !ingress_metadata_t
        p4hir.match_key #exact %vrf : !b12i annotations {name = "meta.ingress_metadata.vrf"}
// CHECK:        %[[REF3:.*]] = bmv2ir.symbol_ref @ingress_metadata_t : !p4hir.ref<!ingress_metadata_t>
// CHECK:        %[[READ3:.*]] = p4hir.read %[[REF3]] : <!ingress_metadata_t>
// CHECK:        %[[VAL:.*]] = p4hir.struct_extract %[[READ3]]["vrf"] : !ingress_metadata_t
// CHECK:        p4hir.match_key #exact %[[VAL]] : !b12i annotations {name = "meta.ingress_metadata.vrf"}
        %val_0 = p4hir.read %arg3 : <!headers>
        %ipv4 = p4hir.struct_extract %val_0["ipv4"] : !headers
// CHECK:        %[[REF4:.*]] = bmv2ir.symbol_ref @headers_ipv4 : !p4hir.ref<!ipv4_t>
// CHECK:        %[[VAL2:.*]] = p4hir.read %[[REF4]] : <!ipv4_t>
// CHECK:        %{{.*}} = p4hir.struct_extract %[[VAL2]]["dstAddr"] : !ipv4_t
        %dstAddr = p4hir.struct_extract %ipv4["dstAddr"] : !ipv4_t
        p4hir.match_key #exact %dstAddr : !b32i annotations {name = "hdr.ipv4.dstAddr"}
      }
      %size = p4hir.table_size #int32768_infint
      p4hir.table_default_action {
        p4hir.call @egress::@NoAction_2 () : () -> ()
      }
    }
    p4hir.control_apply {
      %rewrite_mac_0_apply_result = p4hir.table_apply @egress::@rewrite_mac_0 with key(%arg1) : (!p4hir.ref<!ingress_metadata_t>) -> !rewrite_mac_0
      %res2 = p4hir.table_apply @egress::@ipv4_fib_0 with key(%arg0, %arg1) : (!p4hir.ref<!headers>, !p4hir.ref<!ingress_metadata_t>) -> !rewrite_mac_0
    }
  }
  p4hir.parser @prs(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"} )() {
    p4hir.state @start {
// CHECK:  bmv2ir.symbol_ref @headers_ethernet : !p4hir.ref<!ethernet_t>
      %ethernet_field_ref = p4hir.struct_field_ref %arg1["ethernet"] : <!headers>
      p4hir.transition to @prs::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs::@start
  }
}
