// RUN: p4mlir-opt -p='builtin.module(p4hir-flatten-structs)' %s --split-input-file | FileCheck %s
!anon = !p4hir.enum<on_miss_4, fib_hit_nexthop_1, NoAction_6>
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
!packet_in = !p4hir.extern<"packet_in">
!packet_out = !p4hir.extern<"packet_out">
!type_H = !p4hir.type_var<"H">
!type_M = !p4hir.type_var<"M">
!validity_bit = !p4hir.validity.bit
#exact = #p4hir.match_kind<"exact">
#lpm = #p4hir.match_kind<"lpm">
#undir = #p4hir<dir undir>
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!ethernet_t = !p4hir.header<"ethernet_t", dstAddr: !b48i, srcAddr: !b48i, etherType: !b16i, __valid: !validity_bit>
!ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd: !b16i, nexthop_index: !b16i>
!ipv4_fib_lpm_0 = !p4hir.struct<"ipv4_fib_lpm_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!ipv4_t = !p4hir.header<"ipv4_t", version: !b4i, ihl: !b4i, diffserv: !b8i, totalLen: !b16i, identification: !b16i, flags: !b3i, fragOffset: !b13i, ttl: !b8i, protocol: !b8i, hdrChecksum: !b16i, srcAddr: !b32i, dstAddr: !b32i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int-1_b8i = #p4hir.int<255> : !b8i
#int16384_infint = #p4hir.int<16384> : !infint
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!headers = !p4hir.struct<"headers", ethernet: !ethernet_t, ipv4: !ipv4_t>
!metadata = !p4hir.struct<"metadata", ingress_metadata: !ingress_metadata_t>
// CHECK: ![[METADATA_TY:.*]] = !p4hir.struct<"metadata", ingress_metadata_vrf: !b12i, ingress_metadata_bd: !b16i, ingress_metadata_nexthop_index: !b16i>
!DeparserImpl = !p4hir.control<"DeparserImpl", (!packet_out, !headers)>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!ParserImpl = !p4hir.parser<"ParserImpl", (!packet_in, !p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
!computeChecksum = !p4hir.control<"computeChecksum", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>)>
// CHECK: ![[EGRESS_TY:.*]] = !p4hir.control<"egress", (!p4hir.ref<!{{.*}}>, !p4hir.ref<![[METADATA_TY]]>, !p4hir.ref<!{{.*}}>)>
!egress = !p4hir.control<"egress", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
// CHECK: ![[INGRESS_TY:.*]] = !p4hir.control<"ingress", (!p4hir.ref<!{{.*}}>, !p4hir.ref<![[METADATA_TY]]>, !p4hir.ref<!{{.*}}>)>
!ingress = !p4hir.control<"ingress", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
!verifyChecksum = !p4hir.control<"verifyChecksum", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>)>
module {
  p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "p"}, "vr" : !VerifyChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "vr"}, "ig" : !Ingress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ig"}, "eg" : !Egress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "eg"}, "ck" : !ComputeChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ck"}, "dep" : !Deparser_type_H {p4hir.dir = #undir, p4hir.param_name = "dep"})
  // CHECK: p4hir.parser @ParserImpl(%{{.*}}: {{.*}}, %{{.*}}: {{.*}}, %{{.*}}: !p4hir.ref<![[METADATA_TY]]> {{.*}}, %{{.*}}: !{{.*}})() {
  p4hir.parser @ParserImpl(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!metadata> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_metadata"})() {
    p4hir.state @start {
      p4hir.transition to @ParserImpl::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @ParserImpl::@start
  }
  // CHECK: p4hir.control @egress(%{{.*}}: {{.*}}, %{{.*}}: !p4hir.ref<![[METADATA_TY]]> {{.*}}, %{{.*}}: {{.*}})() {
  p4hir.control @egress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_metadata"})() {
    p4hir.control_apply {
    }
  }
  // CHECK: p4hir.control @ingress(%{{.*}}: {{.*}}, %[[ARG1:.*]]: !p4hir.ref<![[METADATA_TY]]> {{.*}}, %{{.*}}: {{.*}})() {
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "standard_metadata"})() {
    p4hir.control_local @__local_ingress_hdr_0 = %arg0 : !p4hir.ref<!headers>
    // CHECK: p4hir.control_local @__local_ingress_meta_0 = %[[ARG1]] : !p4hir.ref<![[METADATA_TY]]>
    p4hir.control_local @__local_ingress_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_local @__local_ingress_standard_metadata_0 = %arg2 : !p4hir.ref<!standard_metadata_t>
    p4hir.func action @NoAction_6() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @fib_hit_nexthop_1(%arg3: !b16i {p4hir.annotations = {name = "nexthop_index"}, p4hir.dir = #undir, p4hir.param_name = "nexthop_index_2"}) annotations {name = "ingress.fib_hit_nexthop"} {
      // CHECK: %[[FIELD_REF:.*]] = p4hir.struct_field_ref %{{.*}}["ingress_metadata_nexthop_index"] : <!metadata>
      %__local_ingress_meta_0 = p4hir.symbol_ref @ingress::@__local_ingress_meta_0 : !p4hir.ref<!metadata>
      %ingress_metadata_field_ref = p4hir.struct_field_ref %__local_ingress_meta_0["ingress_metadata"] : <!metadata>
      %nexthop_index_field_ref = p4hir.struct_field_ref %ingress_metadata_field_ref["nexthop_index"] : <!ingress_metadata_t>
      // CHECK: p4hir.assign %{{.*}}, %[[FIELD_REF]] : <!b16i>
      p4hir.assign %arg3, %nexthop_index_field_ref : <!b16i>
      p4hir.return
    }
    p4hir.table @ipv4_fib_lpm_0 annotations {name = "ingress.ipv4_fib_lpm"} {
      p4hir.table_actions {
        p4hir.table_action @fib_hit_nexthop_1(%arg3: !b16i {p4hir.annotations = {}, p4hir.param_name = "nexthop_index_2"}) {
          p4hir.call @ingress::@fib_hit_nexthop_1 (%arg3) : (!b16i) -> ()
        }
        p4hir.table_action @NoAction_6() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_6 () : () -> ()
        }
      }
      // CHECK: p4hir.table_key(%[[TABLE_KEY_ARG:.*]]: !p4hir.ref<![[METADATA_TY]]>, %{{.*}}: {{.*}}) {
      p4hir.table_key(%arg3: !p4hir.ref<!metadata>, %arg4: !p4hir.ref<!headers>) {
        %val = p4hir.read %arg3 : <!metadata>
        // CHECK: %[[VAL:.*]] = p4hir.read %[[TABLE_KEY_ARG]] : <![[METADATA_TY]]>
        // CHECK: %[[EXTRACT:.*]] = p4hir.struct_extract %[[VAL:.*]]["ingress_metadata_vrf"] : !metadata
        // CHECK: p4hir.match_key #exact %[[EXTRACT]] : !b12i annotations {name = "meta.ingress_metadata.vrf"}
        %ingress_metadata = p4hir.struct_extract %val["ingress_metadata"] : !metadata
        %vrf = p4hir.struct_extract %ingress_metadata["vrf"] : !ingress_metadata_t
        p4hir.match_key #exact %vrf : !b12i annotations {name = "meta.ingress_metadata.vrf"}
      }
      %size = p4hir.table_size #int16384_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_6 () : () -> ()
      }
    }
    p4hir.control_apply {
      %ipv4_fib_lpm_0_apply_result = p4hir.table_apply @ingress::@ipv4_fib_lpm_0 with key(%arg1, %arg0) : (!p4hir.ref<!metadata>, !p4hir.ref<!headers>) -> !ipv4_fib_lpm_0
    }
  }
  p4hir.control @DeparserImpl(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !headers {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_local @__local_DeparserImpl_packet_0 = %arg0 : !packet_out
    p4hir.control_local @__local_DeparserImpl_hdr_0 = %arg1 : !headers
    p4hir.control_apply {
    }
  }
  p4hir.control @verifyChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"})() {
    p4hir.control_local @__local_verifyChecksum_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_verifyChecksum_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_apply {
    }
  }
  p4hir.control @computeChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"})() {
    p4hir.control_local @__local_computeChecksum_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_computeChecksum_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_apply {
    }
  }
  %ParserImpl = p4hir.construct @ParserImpl () : !ParserImpl
  // CHECK: %[[INGRESS:.*]] = p4hir.construct @ingress () : ![[INGRESS_TY]]
  %ingress = p4hir.construct @ingress () : !ingress
  %verifyChecksum = p4hir.construct @verifyChecksum () : !verifyChecksum
  // CHECK: %[[EGRESS:.*]] = p4hir.construct @egress () : ![[EGRESS_TY]]
  %egress = p4hir.construct @egress () : !egress
  %computeChecksum = p4hir.construct @computeChecksum () : !computeChecksum
  %DeparserImpl = p4hir.construct @DeparserImpl () : !DeparserImpl
  // CHECK: p4hir.instantiate @V1Switch<[!{{.*}}, ![[METADATA_TY]]]> (%{{.*}}, %{{.*}}, %[[INGRESS]], %[[EGRESS]], %{{.*}}, %{{.*}} : !{{.*}}, !{{.*}}, ![[INGRESS_TY]], ![[EGRESS_TY]], !{{.*}}, !{{.*}}) as @main
  p4hir.instantiate @V1Switch<[!headers, !metadata]> (%ParserImpl, %verifyChecksum, %ingress, %egress, %computeChecksum, %DeparserImpl : !ParserImpl, !verifyChecksum, !ingress, !egress, !computeChecksum, !DeparserImpl) as @main
}

// -----

!b16i = !p4hir.bit<16>
!ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b16i, bd: !b16i, nexthop_index: !b16i>
!metadata = !p4hir.struct<"metadata", ingress_metadata: !ingress_metadata_t>
// CHECK: !metadata = !p4hir.struct<"metadata", ingress_metadata_vrf: !b16i, ingress_metadata_bd: !b16i, ingress_metadata_nexthop_index: !b16i>
module {

// CHECK-LABEL:   p4hir.func action @field_ref_tree(
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.ref<!metadata>,
// CHECK-SAME:      %[[ARG1:.*]]: !b16i) {
// CHECK-DAG:           %[[VAL_0:.*]] = p4hir.struct_field_ref %[[ARG0]]["ingress_metadata_vrf"] : <!metadata>
// CHECK-DAG:           %[[VAL_1:.*]] = p4hir.struct_field_ref %[[ARG0]]["ingress_metadata_nexthop_index"] : <!metadata>
// CHECK:               p4hir.assign %[[ARG1]], %[[VAL_1]] : <!b16i>
// CHECK:               p4hir.assign %[[ARG1]], %[[VAL_0]] : <!b16i>
// CHECK:               p4hir.return
// CHECK:         }


  p4hir.func action @field_ref_tree(%arg1: !p4hir.ref<!metadata>, %arg2 : !b16i) {
      %ingress_metadata_field_ref = p4hir.struct_field_ref %arg1["ingress_metadata"] : <!metadata>
      %nexthop_index_field_ref = p4hir.struct_field_ref %ingress_metadata_field_ref["nexthop_index"] : <!ingress_metadata_t>
      p4hir.assign %arg2, %nexthop_index_field_ref : <!b16i>
      %vrf_index_field_ref = p4hir.struct_field_ref %ingress_metadata_field_ref["vrf"] : <!ingress_metadata_t>
      p4hir.assign %arg2, %vrf_index_field_ref : <!b16i>
      p4hir.return
  }

// CHECK-LABEL:   p4hir.func action @extract_tree(
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.ref<!metadata>,
// CHECK-SAME:      %[[ARG1:.*]]: !p4hir.ref<!b16i>,
// CHECK-SAME:      %[[ARG2:.*]]: !p4hir.ref<!b16i>) {
// CHECK-DAG:           %[[VAL_0:.*]] = p4hir.read %[[ARG0]] : <!metadata>
// CHECK-DAG:           %[[VAL_1:.*]] = p4hir.struct_extract %[[VAL_0]]["ingress_metadata_vrf"] : !metadata
// CHECK-DAG:           %[[VAL_2:.*]] = p4hir.struct_extract %[[VAL_0]]["ingress_metadata_nexthop_index"] : !metadata
// CHECK:               p4hir.assign %[[VAL_2]], %[[ARG1]] : <!b16i>
// CHECK:               p4hir.assign %[[VAL_1]], %[[ARG2]] : <!b16i>
// CHECK:               p4hir.return
// CHECK:         }

  p4hir.func action @extract_tree(%arg1: !p4hir.ref<!metadata>, %arg2 : !p4hir.ref<!b16i>, %arg3 : !p4hir.ref<!b16i>) {
    %val = p4hir.read %arg1 : <!metadata>
    %ingress_metadata = p4hir.struct_extract %val["ingress_metadata"] : !metadata
    %nexthop_index = p4hir.struct_extract %ingress_metadata["nexthop_index"] : !ingress_metadata_t
    p4hir.assign %nexthop_index, %arg2 : <!b16i>
    %vrf = p4hir.struct_extract %ingress_metadata["vrf"] : !ingress_metadata_t
    p4hir.assign %vrf, %arg3 : <!b16i>
    p4hir.return
  }

// CHECK-LABEL:   p4hir.func action @mixed(
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.ref<!metadata>,
// CHECK-SAME:      %[[ARG1:.*]]: !p4hir.ref<!b16i>,
// CHECK-SAME:      %[[ARG2:.*]]: !p4hir.ref<!b16i>) {
// CHECK-DAG:           %[[VAL_0:.*]] = p4hir.struct_field_ref %[[ARG0]]["ingress_metadata_vrf"] : <!metadata>
// CHECK-DAG:           %[[VAL_1:.*]] = p4hir.read %[[VAL_0]] : <!b16i>
// CHECK-DAG:           %[[VAL_2:.*]] = p4hir.struct_field_ref %[[ARG0]]["ingress_metadata_nexthop_index"] : <!metadata>
// CHECK-DAG:           %[[VAL_3:.*]] = p4hir.read %[[VAL_2]] : <!b16i>
// CHECK:               p4hir.assign %[[VAL_3]], %[[ARG1]] : <!b16i>
// CHECK:               p4hir.assign %[[VAL_1]], %[[ARG2]] : <!b16i>
// CHECK:               p4hir.return
// CHECK:         }

  p4hir.func action @mixed(%arg1: !p4hir.ref<!metadata>, %arg2 : !p4hir.ref<!b16i>, %arg3 : !p4hir.ref<!b16i>) {
    %ingress_metadata_ref = p4hir.struct_field_ref %arg1["ingress_metadata"] : <!metadata>
    %ingress_metadata = p4hir.read %ingress_metadata_ref : <!ingress_metadata_t>
    %nexthop_index = p4hir.struct_extract %ingress_metadata["nexthop_index"] : !ingress_metadata_t
    p4hir.assign %nexthop_index, %arg2 : <!b16i>
    %vrf = p4hir.struct_extract %ingress_metadata["vrf"] : !ingress_metadata_t
    p4hir.assign %vrf, %arg3 : <!b16i>
    p4hir.return
  }

}

// -----

!validity_bit = !p4hir.validity.bit
!b8i = !p4hir.bit<8>
!switch_metadata_t = !p4hir.struct<"switch_metadata_t", port1: !b8i, port2: !b8i>
!serialized_switch_metadata_t = !p4hir.header<"serialized_switch_metadata_t", meta: !switch_metadata_t, __valid: !validity_bit>
// CHECK: !serialized_switch_metadata_t = !p4hir.header<"serialized_switch_metadata_t", meta_port1: !b8i, meta_port2: !b8i, __valid: !validity_bit>
!parsed_packet_t = !p4hir.struct<"parsed_packet_t", mirrored_md: !serialized_switch_metadata_t>
#valid = #p4hir<validity.bit valid> : !validity_bit
#int0_b8i = #p4hir.int<0> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
module {

// CHECK: #[[$ATTR_3:.+]] = #p4hir.int<0> : !b8i
// CHECK: #[[$ATTR_4:.+]] = #p4hir.int<1> : !b8i
// CHECK: #[[$ATTR_5:.+]] = #p4hir<validity.bit valid> : !validity_bit
// CHECK-LABEL:   p4hir.func action @headers_with_struct(
// CHECK-SAME:      %[[ARG0:.*]]: !p4hir.ref<!parsed_packet_t>) {
// CHECK-DAG:           %[[VAL_0:.*]] = p4hir.struct_field_ref %[[ARG0]]["mirrored_md"] : <!parsed_packet_t>
// CHECK-DAG:           %[[VAL_1:.*]] = p4hir.const #[[$ATTR_5]]
// CHECK-DAG:           %[[VAL_2:.*]] = p4hir.struct_field_ref %[[VAL_0]]["__valid"] : <!serialized_switch_metadata_t>
// CHECK:               p4hir.assign %[[VAL_1]], %[[VAL_2]] : <!validity_bit>
// CHECK-DAG:           %[[VAL_3:.*]] = p4hir.struct_field_ref %[[ARG0]]["mirrored_md"] : <!parsed_packet_t>
// CHECK-DAG:           %[[VAL_4:.*]] = p4hir.struct_field_ref %[[VAL_3]]["meta_port2"] : <!serialized_switch_metadata_t>
// CHECK-DAG:           %[[VAL_5:.*]] = p4hir.struct_field_ref %[[VAL_3]]["meta_port1"] : <!serialized_switch_metadata_t>
// CHECK-DAG:           %[[VAL_6:.*]] = p4hir.const #[[$ATTR_3]]
// CHECK-DAG:           %[[VAL_7:.*]] = p4hir.const #[[$ATTR_4]]
// CHECK:               p4hir.assign %[[VAL_6]], %[[VAL_5]] : <!b8i>
// CHECK:               p4hir.assign %[[VAL_7]], %[[VAL_4]] : <!b8i>
// CHECK:               p4hir.return
// CHECK:         }


  p4hir.func action @headers_with_struct(%arg0: !p4hir.ref<!parsed_packet_t>) {
    %mirrored_md_field_ref = p4hir.struct_field_ref %arg0["mirrored_md"] : <!parsed_packet_t>
    %valid = p4hir.const #valid
    %__valid_field_ref = p4hir.struct_field_ref %mirrored_md_field_ref["__valid"] : <!serialized_switch_metadata_t>
    p4hir.assign %valid, %__valid_field_ref : <!validity_bit>
    %mirrored_md_field_ref_0 = p4hir.struct_field_ref %arg0["mirrored_md"] : <!parsed_packet_t>
    %meta_field_ref = p4hir.struct_field_ref %mirrored_md_field_ref_0["meta"] : <!serialized_switch_metadata_t>
    %c0_b8i = p4hir.const #int0_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %port1_field_ref = p4hir.struct_field_ref %meta_field_ref["port1"] : <!switch_metadata_t>
    %port2_field_ref = p4hir.struct_field_ref %meta_field_ref["port2"] : <!switch_metadata_t>
    p4hir.assign %c0_b8i, %port1_field_ref : <!b8i>
    p4hir.assign %c1_b8i, %port2_field_ref : <!b8i>
    p4hir.return
  }
}
