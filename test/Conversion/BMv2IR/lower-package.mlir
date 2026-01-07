// RUN: p4mlir-opt -p='builtin.module(lower-package)' %s | FileCheck %s
!CloneType = !p4hir.enum<"CloneType", I2E, E2E>
!CounterType = !p4hir.enum<"CounterType", packets, bytes, packets_and_bytes>
!HashAlgorithm = !p4hir.enum<"HashAlgorithm", crc32, crc32_custom, crc16, crc16_custom, random, identity, csum16, xor16>
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
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!H = !p4hir.header<"H", a: !b8i, b: !b8i, __valid: !validity_bit>
!ethernet_t = !p4hir.header<"ethernet_t", dst_addr: !b48i, src_addr: !b48i, eth_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int-1_b16i = #p4hir.int<65535> : !b16i
#int-1_b48i = #p4hir.int<281474976710655> : !b48i
#int10_b8i = #p4hir.int<10> : !b8i
#int11_b8i = #p4hir.int<11> : !b8i
#int255_b9i = #p4hir.int<255> : !b9i
#valid = #p4hir<validity.bit valid> : !validity_bit
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!headers = !p4hir.struct<"headers", eth_hdr: !ethernet_t, h: !H>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!deparser = !p4hir.control<"deparser", (!packet_out, !headers)>
!egress = !p4hir.control<"egress", (!p4hir.ref<!headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t>)>
!ingress = !p4hir.control<"ingress", (!p4hir.ref<!headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t>)>
!p = !p4hir.parser<"p", (!packet_in, !p4hir.ref<!headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t>)>
!update = !p4hir.control<"update", (!p4hir.ref<!headers>, !p4hir.ref<!Meta>)>
!vrfy = !p4hir.control<"vrfy", (!p4hir.ref<!headers>, !p4hir.ref<!Meta>)>
module {
  p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "p"}, "vr" : !VerifyChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "vr"}, "ig" : !Ingress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ig"}, "eg" : !Egress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "eg"}, "ck" : !ComputeChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ck"}, "dep" : !Deparser_type_H {p4hir.dir = #undir, p4hir.param_name = "dep"})
  p4hir.parser @p(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.state @start {
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @i(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @vrfy(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @update(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @e(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @dep(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !headers {p4hir.dir = #in, p4hir.param_name = "h"})() {
    p4hir.control_apply {
    }
  }
  %p = p4hir.construct @p () : !p
  %vrfy = p4hir.construct @vrfy () : !vrfy
  %ingress = p4hir.construct @i() : !ingress
  %egress = p4hir.construct @e() : !egress
  %update = p4hir.construct @update () : !update
  %deparser = p4hir.construct @dep() : !deparser
// CHECK-NOT p4hir.construct
  p4hir.instantiate @V1Switch<[!headers, !Meta]> (%p, %vrfy, %ingress, %egress, %update, %deparser : !p, !vrfy, !ingress, !egress, !update, !deparser) as @main
// CHECK: bmv2ir.v1switch @main parser @parser, verify_checksum @vrfy, ingress @ingress, egress @egress, compute_checksum @update, deparser @deparser
}
