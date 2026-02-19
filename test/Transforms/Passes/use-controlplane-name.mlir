// RUN: p4mlir-opt -p='builtin.module(p4hir-use-controlplane-names)' %s | FileCheck %s
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
!packet_in = !p4hir.extern<"packet_in">
!packet_out = !p4hir.extern<"packet_out">
!string = !p4hir.string
!type_D = !p4hir.type_var<"D">
!type_H = !p4hir.type_var<"H">
!type_M = !p4hir.type_var<"M">
!type_O = !p4hir.type_var<"O">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#exact = #p4hir.match_kind<"exact">
#in = #p4hir<dir in>
#inout = #p4hir<dir inout>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!data_t = !p4hir.header<"data_t", c1: !b8i, c2: !b8i, c3: !b8i, r1: !b32i, r2: !b32i, r3: !b32i, b1: !b8i, b2: !b8i, b3: !b8i, b4: !b8i, b5: !b8i, b6: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
!t1_0 = !p4hir.struct<"t1_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!t2_0 = !p4hir.struct<"t2_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon1>
!t3_0 = !p4hir.struct<"t3_0", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon2>
#int0_b8i = #p4hir.int<0> : !b8i
#int1024_infint = #p4hir.int<1024> : !infint
!Headers = !p4hir.struct<"Headers", data: !data_t>
!ingress = !p4hir.control<"ingress", (!p4hir.ref<!Headers>, !p4hir.ref<!Meta>, !p4hir.ref<!standard_metadata_t>)>
module {
  p4hir.control @ingress(%arg0: !p4hir.ref<!Headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!Meta> {p4hir.dir = #inout, p4hir.param_name = "m"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "s"})() {
    p4hir.control_local @__local_ingress_hdr_0 = %arg0 : !p4hir.ref<!Headers>
    p4hir.control_local @__local_ingress_m_0 = %arg1 : !p4hir.ref<!Meta>
    p4hir.control_local @__local_ingress_s_0 = %arg2 : !p4hir.ref<!standard_metadata_t>
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
// CHECK: p4hir.table @ingress.t1 annotations {name = "ingress.t1"}
    p4hir.table @t1_0 annotations {name = "ingress.t1"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %val = p4hir.read %arg3 : <!Headers>
        %data = p4hir.struct_extract %val["data"] : !Headers
        %r1 = p4hir.struct_extract %data["r1"] : !data_t
        p4hir.match_key #exact %r1 : !b32i annotations {name = "hdr.data.r1"}
      }
      p4hir.table_actions {
        p4hir.table_action @noop() {
          p4hir.call @ingress::@noop () : () -> ()
        }
      }
      %size = p4hir.table_size #int1024_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@noop () : () -> ()
      }
    }
// CHECK: p4hir.table @ingress.t2 annotations {name = "ingress.t2"}
    p4hir.table @t2_0 annotations {name = "ingress.t2"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %val = p4hir.read %arg3 : <!Headers>
        %data = p4hir.struct_extract %val["data"] : !Headers
        %r2 = p4hir.struct_extract %data["r2"] : !data_t
        p4hir.match_key #exact %r2 : !b32i annotations {name = "hdr.data.r2"}
      }
      p4hir.table_actions {
        p4hir.table_action @noop_1() {
          p4hir.call @ingress::@noop_1 () : () -> ()
        }
      }
      %size = p4hir.table_size #int1024_infint
      p4hir.table_default_action {
        p4hir.call @ingress::@noop_1 () : () -> ()
      }
    }
// CHECK: p4hir.table @ingress.t3 annotations {name = "ingress.t3"}
    p4hir.table @t3_0 annotations {name = "ingress.t3"} {
      p4hir.table_key(%arg3: !p4hir.ref<!Headers>) {
        %val = p4hir.read %arg3 : <!Headers>
        %data = p4hir.struct_extract %val["data"] : !Headers
        %b1 = p4hir.struct_extract %data["b1"] : !data_t
        p4hir.match_key #exact %b1 : !b8i annotations {name = "hdr.data.b1"}
      }
      p4hir.table_actions {
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
      %val = p4hir.read %arg0 : <!Headers>
      %data = p4hir.struct_extract %val["data"] : !Headers
      %c1 = p4hir.struct_extract %data["c1"] : !data_t
      %c0_b8i = p4hir.const #int0_b8i
      %eq = p4hir.cmp(eq, %c1 : !b8i, %c0_b8i : !b8i)
      p4hir.if %eq {
// CHECK: p4hir.table_apply @ingress::@ingress.t1 
        %t1_0_apply_result = p4hir.table_apply @ingress::@t1_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t1_0
        %miss = p4hir.struct_extract %t1_0_apply_result["miss"] : !t1_0
        p4hir.if %miss {
// CHECK: p4hir.table_apply @ingress::@ingress.t2
          %t2_0_apply_result = p4hir.table_apply @ingress::@t2_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t2_0
        }
// CHECK: p4hir.table_apply @ingress::@ingress.t3
        %t3_0_apply_result = p4hir.table_apply @ingress::@t3_0 with key(%arg0) : (!p4hir.ref<!Headers>) -> !t3_0
      }
    }
  }
}
