// RUN: p4mlir-opt --set-corelib %s | FileCheck %s
!Meta = !p4hir.struct<"Meta">
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
// CHECK: !packet_in = !p4hir.extern<"packet_in" annotations {corelib = true}>
// CHECK: !packet_out = !p4hir.extern<"packet_out" annotations {corelib = true}>
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
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!H = !p4hir.header<"H", a: !b8i, b: !b8i, __valid: !validity_bit>
!ethernet_t = !p4hir.header<"ethernet_t", dst_addr: !b48i, src_addr: !b48i, eth_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#valid = #p4hir<validity.bit valid> : !validity_bit
!headers = !p4hir.struct<"headers", eth_hdr: !ethernet_t, h: !H>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!deparser = !p4hir.control<"deparser", (!packet_out, !headers)>
!p = !p4hir.parser<"p", (!packet_in, !p4hir.ref<!headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t>)>
module {
// CHECK: p4hir.extern @packet_in annotations {corelib = true} {
// CHECK:   p4hir.overload_set @extract {
// CHECK:     p4hir.func @extract_0{{.*}} annotations {corelib = true}
// CHECK:     p4hir.func @extract_1{{.*}}annotations {corelib = true}
  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @extract_1<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }
// CHECK: p4hir.extern @packet_out annotations {corelib = true} {
// CHECK:   p4hir.func @emit{{.*}}{corelib = true}
  p4hir.extern @packet_out {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  p4hir.parser @p(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "sm"})() {
    p4hir.state @start {
      p4hir.scope {
        %eth_hdr_field_ref = p4hir.struct_field_ref %arg1["eth_hdr"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!ethernet_t>
        p4hir.call_method @packet_in::@extract<[!ethernet_t]> (%hdr_out_arg) of %arg0 : !packet_in : (!p4hir.ref<!ethernet_t>) -> ()
// CHECK: p4hir.call_method @packet_in::@extract{{.*}} of %{{.*}} : !packet_in
        %val_0 = p4hir.read %hdr_out_arg : <!ethernet_t>
        p4hir.assign %val_0, %eth_hdr_field_ref : <!ethernet_t>
      }

      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @deparser(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !headers {p4hir.dir = #in, p4hir.param_name = "h"})() {
    p4hir.control_local @__local_deparser_pkt_0 = %arg0 : !packet_out
    p4hir.control_local @__local_deparser_h_0 = %arg1 : !headers
    p4hir.control_apply {
      p4hir.call_method @packet_out::@emit<[!headers]> (%arg1) of %arg0 : !packet_out : (!headers) -> ()
// CHECK: p4hir.call_method @packet_out::@emit{{.*}} of %{{.*}} : !packet_out
    }
  }
}
