// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir,canonicalize)' %s --split-input-file | FileCheck %s
!Meta = !p4hir.struct<"Meta">
!anon = !p4hir.enum<setb1, noop>
!anon1 = !p4hir.enum<setb2, noop_1>
!anon2 = !p4hir.enum<setb3, noop_2, NoAction_1>
!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!validity_bit = !p4hir.validity.bit
#exact = #p4hir.match_kind<"exact">
!data_t = !p4hir.header<"data_t", c1: !b8i, c2: !b8i, c3: !b8i, r1: !b32i, r2: !b32i, r3: !b32i, b1: !b8i, b2: !b8i, b3: !b8i, b4: !b8i, b5: !b8i, b6: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t1_0 = !p4hir.struct<"t1_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!t2_0 = !p4hir.struct<"t2_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon1>
!t3_0 = !p4hir.struct<"t3_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon2>
#int0_b8i = #p4hir.int<0> : !b8i
#int1024_infint = #p4hir.int<1024> : !infint
!Headers = !p4hir.struct<"Headers", data: !data_t>
module {
  bmv2ir.header_instance @hdr_out_arg_var_0 : !p4hir.ref<!data_t>
  bmv2ir.header_instance @Headers_data : !p4hir.ref<!data_t>
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!Headers> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "h"}, %arg2: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "stdmeta"})() {
    p4hir.state @start {
      p4hir.transition to @parser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @parser::@start
  }
  p4hir.control @vrfy(%arg0: !p4hir.ref<!Headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"})() {
    p4hir.control_apply {
    }
  }
  
// CHECK:    bmv2ir.table @t1_0
// CHECK:     actions [@ingress::@setb1, @ingress::@noop]
// CHECK:     next_tables #bmv2ir.hit_miss<hit @ingress::@t3_0 miss @ingress::@t2_0>
// CHECK:     type  simple
// CHECK:     match_type  exact
// CHECK:     keys [#bmv2ir.table_key<type exact, header @Headers_data["r1"]>]
// CHECK:     support_timeout false
// CHECK:     default_entry <action @ingress::@noop, action_const true, action_entry_const true>
// CHECK:     size 1024
// CHECK:    bmv2ir.table @t2_0
// CHECK:     actions [@ingress::@setb2, @ingress::@noop_1]
// CHECK:     next_tables [#bmv2ir.action_table<@ingress::@setb2 : @ingress::@t3_0>, #bmv2ir.action_table<@ingress::@noop_1 : @ingress::@t3_0>]
// CHECK:     type  simple
// CHECK:     match_type  exact
// CHECK:     keys [#bmv2ir.table_key<type exact, header @Headers_data["r2"]>]
// CHECK:     support_timeout false
// CHECK:     default_entry <action @ingress::@noop_1, action_const true, action_entry_const true>
// CHECK:     size 1024
// CHECK:    bmv2ir.table @t3_0
// CHECK:     actions [@ingress::@setb3, @ingress::@noop_2, @ingress::@NoAction_1]
// CHECK:     next_tables [#bmv2ir.action_table<@ingress::@setb3>, #bmv2ir.action_table<@ingress::@noop_2>, #bmv2ir.action_table<@ingress::@NoAction_1>]
// CHECK:     type  simple
// CHECK:     match_type  exact
// CHECK:     keys [#bmv2ir.table_key<type exact, header @Headers_data["b1"]>, #bmv2ir.table_key<type exact, header @Headers_data["b2"]>]
// CHECK:     support_timeout false
// CHECK:     default_entry <action @ingress::@NoAction_1, action_const true, action_entry_const true>
// CHECK:     size 1024
// CHECK:    bmv2ir.conditional @cond_node_0 then @ingress::@t1_0 expr {
// CHECK:      %0 = bmv2ir.field @Headers_data["c1"] -> !b8i
// CHECK:      %eq = p4hir.cmp(eq, %0 : !b8i, %c0_b8i : !b8i)
// CHECK:      bmv2ir.yield %eq : !p4hir.bool

  p4hir.control @ingress(%arg0: !p4hir.ref<!Headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "s"})() {
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.func action @NoAction_1() annotations {name = ".NoAction", noWarn = "unused"} {
      p4hir.return
    }
    p4hir.func action @noop() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @noop_1() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @noop_2() annotations {name = "ingress.noop"} {
      p4hir.return
    }
    p4hir.func action @setb1(%arg3: !b8i {p4hir.annotations = {name = "b1"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b1_1"}) annotations {name = "ingress.setb1"} {
      %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
      %b1_field_ref = p4hir.struct_field_ref %Headers_data["b1"] : <!data_t>
      p4hir.assign %arg3, %b1_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.func action @setb2(%arg3: !b8i {p4hir.annotations = {name = "b2"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b2_1"}) annotations {name = "ingress.setb2"} {
      %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
      %b2_field_ref = p4hir.struct_field_ref %Headers_data["b2"] : <!data_t>
      p4hir.assign %arg3, %b2_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.func action @setb3(%arg3: !b8i {p4hir.annotations = {name = "b3"}, p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "b3_1"}) annotations {name = "ingress.setb3"} {
      %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
      %b3_field_ref = p4hir.struct_field_ref %Headers_data["b3"] : <!data_t>
      p4hir.assign %arg3, %b3_field_ref : <!b8i>
      p4hir.return
    }
    p4hir.table @t1_0 annotations {name = "ingress.t1"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
        %r1_field_ref = p4hir.struct_field_ref %Headers_data["r1"] : <!data_t>
        %val = p4hir.read %r1_field_ref : <!b32i>
        p4hir.match_key #exact %val : !b32i annotations {name = "hdr.data.r1"}
      }
      p4hir.table_actions {
        p4hir.table_action @setb1(%arg3: !b8i {p4hir.annotations = {}, p4hir.param_name = "b1_1"}) {
          p4hir.call @ingress::@setb1 (%arg3) : (!b8i) -> ()
        }
        p4hir.table_action @noop() {
          p4hir.call @ingress::@noop () : () -> ()
        }
      }
      %size = p4hir.table_size #int1024_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@noop () : () -> ()
      }
    }
    p4hir.table @t2_0 annotations {name = "ingress.t2"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
        %r2_field_ref = p4hir.struct_field_ref %Headers_data["r2"] : <!data_t>
        %val = p4hir.read %r2_field_ref : <!b32i>
        p4hir.match_key #exact %val : !b32i annotations {name = "hdr.data.r2"}
      }
      p4hir.table_actions {
        p4hir.table_action @setb2(%arg3: !b8i {p4hir.annotations = {}, p4hir.param_name = "b2_1"}) {
          p4hir.call @ingress::@setb2 (%arg3) : (!b8i) -> ()
        }
        p4hir.table_action @noop_1() {
          p4hir.call @ingress::@noop_1 () : () -> ()
        }
      }
      %size = p4hir.table_size #int1024_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@noop_1 () : () -> ()
      }
    }
    p4hir.table @t3_0 annotations {name = "ingress.t3"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
        %b1_field_ref = p4hir.struct_field_ref %Headers_data["b1"] : <!data_t>
        %val = p4hir.read %b1_field_ref : <!b8i>
        p4hir.match_key #exact %val : !b8i annotations {name = "hdr.data.b1"}
        %Headers_data_0 = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
        %b2_field_ref = p4hir.struct_field_ref %Headers_data_0["b2"] : <!data_t>
        %val_1 = p4hir.read %b2_field_ref : <!b8i>
        p4hir.match_key #exact %val_1 : !b8i annotations {name = "hdr.data.b2"}
      }
      p4hir.table_actions {
        p4hir.table_action @setb3(%arg3: !b8i {p4hir.annotations = {}, p4hir.param_name = "b3_1"}) {
          p4hir.call @ingress::@setb3 (%arg3) : (!b8i) -> ()
        }
        p4hir.table_action @noop_2() {
          p4hir.call @ingress::@noop_2 () : () -> ()
        }
        p4hir.table_action @NoAction_1() annotations {defaultonly} {
          p4hir.call @ingress::@NoAction_1 () : () -> ()
        }
      }
      %size = p4hir.table_size #int1024_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@NoAction_1 () : () -> ()
      }
    }
    p4hir.control_apply {
      bmv2ir.if @cond_node_0 expr {
        %Headers_data = bmv2ir.symbol_ref @Headers_data : !p4hir.ref<!data_t>
        %c1_field_ref = p4hir.struct_field_ref %Headers_data["c1"] : <!data_t>
        %val = p4hir.read %c1_field_ref : <!b8i>
        %eq = p4hir.cmp(eq, %val : !b8i, %c0_b8i : !b8i)
        bmv2ir.yield %eq : !p4hir.bool
      } then {
        %t1_0_apply_result = p4hir.table_apply @ingress::@t1_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t1_0
        %miss = p4hir.struct_extract %t1_0_apply_result["miss"] : !t1_0
        p4hir.if %miss {
          %t2_0_apply_result = p4hir.table_apply @ingress::@t2_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t2_0
        }
        %t3_0_apply_result = p4hir.table_apply @ingress::@t3_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t3_0
      } else {
      }
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!Headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "s"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @update(%arg0: !p4hir.ref<!Headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !Headers {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h"})() {
    p4hir.control_apply {
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @vrfy, ingress @ingress, egress @egress, compute_checksum @update, deparser @deparser
}

