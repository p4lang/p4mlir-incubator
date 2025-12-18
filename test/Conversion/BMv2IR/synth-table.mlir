// RUN: p4mlir-opt -p='builtin.module(synth-tables)' %s | FileCheck %s
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
!validity_bit = !p4hir.validity.bit
!H = !p4hir.header<"H", a: !b8i, b: !b8i, __valid: !validity_bit>
!ethernet_t = !p4hir.header<"ethernet_t", dst_addr: !b48i, src_addr: !b48i, eth_type: !b16i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int-1_b16i = #p4hir.int<65535> : !b16i
#int-1_b48i = #p4hir.int<281474976710655> : !b48i
#int10_b8i = #p4hir.int<10> : !b8i
#int11_b8i = #p4hir.int<11> : !b8i
#valid = #p4hir<validity.bit valid> : !validity_bit
// CHECK: #[[ATTR_1024:.*]] = #p4hir.int<1024> : !infint
!headers = !p4hir.struct<"headers", eth_hdr: !ethernet_t, h: !H>
module {
  bmv2ir.header_instance @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
  bmv2ir.header_instance @ingress0_h : !p4hir.ref<!H>
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "sm"})() {
    p4hir.func action @dummy_action_2() {
      %ingress0_eth_hdr = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %src_addr_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr["src_addr"] : <!ethernet_t>
      %c-1_b48i_0 = p4hir.const #int-1_b48i
      p4hir.assign %c-1_b48i_0, %src_addr_field_ref : <!b48i>
      p4hir.return
    }
    p4hir.func action @dummy_action_1() {
      %ingress0_eth_hdr = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %dst_addr_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr["dst_addr"] : <!ethernet_t>
      %c-1_b48i_0 = p4hir.const #int-1_b48i
      p4hir.assign %c-1_b48i_0, %dst_addr_field_ref : <!b48i>
      %ingress0_eth_hdr_1 = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %src_addr_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr_1["src_addr"] : <!ethernet_t>
      p4hir.assign %c-1_b48i_0, %src_addr_field_ref : <!b48i>
      %ingress0_eth_hdr_2 = bmv2ir.symbol_ref @ingress0_eth_hdr : !p4hir.ref<!ethernet_t>
      %eth_type_field_ref = p4hir.struct_field_ref %ingress0_eth_hdr_2["eth_type"] : <!ethernet_t>
      %c-1_b16i_3 = p4hir.const #int-1_b16i
      p4hir.assign %c-1_b16i_3, %eth_type_field_ref : <!b16i>
      p4hir.return
    }
    p4hir.func action @dummy_action_0() {
      %ingress0_h = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
      %a_field_ref = p4hir.struct_field_ref %ingress0_h["a"] : <!H>
      %c10_b8i_0 = p4hir.const #int10_b8i
      p4hir.assign %c10_b8i_0, %a_field_ref : <!b8i>
      %ingress0_h_1 = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
      %b_field_ref = p4hir.struct_field_ref %ingress0_h_1["b"] : <!H>
      %c11_b8i_2 = p4hir.const #int11_b8i
      p4hir.assign %c11_b8i_2, %b_field_ref : <!b8i>
      p4hir.return
    }
    %c11_b8i = p4hir.const #int11_b8i
    %c10_b8i = p4hir.const #int10_b8i
    %c-1_b16i = p4hir.const #int-1_b16i
    %c-1_b48i = p4hir.const #int-1_b48i
// CHECK:    p4hir.table @dummy_table_0 {
// CHECK:      p4hir.table_action @dummy_action_1() {
// CHECK:        p4hir.call @ingress::@dummy_action_1 () : () -> ()
// CHECK:      }
// CHECK:      p4hir.table_default_action const {
// CHECK:        p4hir.call @ingress::@dummy_action_1 () : () -> ()
// CHECK:      }
// CHECK:      %{{.*}} = p4hir.table_size #[[ATTR_1024]]
// CHECK:    }
// CHECK:    p4hir.table @dummy_table_1 {
// CHECK:      p4hir.table_action @dummy_action_0() {
// CHECK:        p4hir.call @ingress::@dummy_action_0 () : () -> ()
// CHECK:      }
// CHECK:      p4hir.table_default_action const {
// CHECK:        p4hir.call @ingress::@dummy_action_0 () : () -> ()
// CHECK:      }
// CHECK:      %{{.*}} = p4hir.table_size #[[ATTR_1024]]
// CHECK:    }
// CHECK:    p4hir.table @dummy_table_2 {
// CHECK:      p4hir.table_action @dummy_action_2() {
// CHECK:        p4hir.call @ingress::@dummy_action_2 () : () -> ()
// CHECK:      }
// CHECK:      p4hir.table_default_action const {
// CHECK:        p4hir.call @ingress::@dummy_action_2 () : () -> ()
// CHECK:      }
// CHECK:      %{{.*}} = p4hir.table_size #[[ATTR_1024]]
// CHECK:    }
    p4hir.control_apply {
      p4hir.call @ingress::@dummy_action_1 () : () -> ()
// CHECK: p4hir.table_apply @ingress::@dummy_table_0 with key() : () -> !dummy_apply_res
      bmv2ir.if @cond_node_0 expr {
        %valid = p4hir.const #valid
        %ingress0_h = bmv2ir.symbol_ref @ingress0_h : !p4hir.ref<!H>
        %__valid_field_ref = p4hir.struct_field_ref %ingress0_h["__valid"] : <!H>
        %val = p4hir.read %__valid_field_ref : <!validity_bit>
        %eq = p4hir.cmp(eq, %val : !validity_bit, %valid : !validity_bit)
        bmv2ir.yield %eq : !p4hir.bool
      } then {
        p4hir.call @ingress::@dummy_action_0 () : () -> ()
// CHECK: p4hir.table_apply @ingress::@dummy_table_1 with key() : () -> !dummy_apply_res
      } else {
      }
      p4hir.call @ingress::@dummy_action_2 () : () -> ()
// CHECK: p4hir.table_apply @ingress::@dummy_table_2 with key() : () -> !dummy_apply_res
    }
  }
}
