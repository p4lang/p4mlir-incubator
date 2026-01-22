// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir,canonicalize)' %s --split-input-file | FileCheck %s
!b16i = !p4hir.bit<16>
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
!header_bottom = !p4hir.header<"header_bottom", length: !b8i, data: !p4hir.varbit<256>, __valid: !validity_bit>
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
!Headers_t = !p4hir.struct<"Headers_t", top: !header_top, one: !header_one, two: !header_two, bottom: !header_bottom>
module {
  bmv2ir.header_instance @prs_e_0 : !p4hir.ref<!header_one>
  bmv2ir.header_instance @prs1_top : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs1_one : !p4hir.ref<!header_one>
  bmv2ir.header_instance @prs1_two : !p4hir.ref<!header_two>
// CHECK:  bmv2ir.header_instance @prs_e_0 : !bmv2ir.header<"header_one", [type:!p4hir.bit<8>, data:!p4hir.bit<8>], max_length = 2>
// CHECK:  bmv2ir.header_instance @prs1_top : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
// CHECK:  bmv2ir.header_instance @prs1_one : !bmv2ir.header<"header_one", [type:!p4hir.bit<8>, data:!p4hir.bit<8>], max_length = 2>
// CHECK:  bmv2ir.header_instance @prs1_two : !bmv2ir.header<"header_two", [type:!p4hir.bit<8>, data:!p4hir.bit<16>], max_length = 3>
  p4hir.parser @prs(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
// CHECK:  bmv2ir.parser @prs init_state @prs::@start {
    %prs_e_0 = bmv2ir.symbol_ref @prs_e_0 : !p4hir.ref<!header_one>
    p4hir.state @start {
      p4hir.scope {
        %prs1_top = bmv2ir.symbol_ref @prs1_top : !p4hir.ref<!header_top>
        p4corelib.extract_header %prs1_top : <!header_top> from %arg0 : !p4corelib.packet_in
      }
      p4hir.transition to @prs::@parse_headers
    }
// CHECK:    bmv2ir.state @start
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_headers
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular @prs1_top
// CHECK:    }
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
// CHECK:    bmv2ir.state @parse_headers
// CHECK:     transition_key {
// CHECK:      bmv2ir.lookahead<0, 8>
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  hexstr value #int1_b8i next_state @prs::@parse_one
// CHECK:      bmv2ir.transition type  hexstr value #int2_b8i next_state @prs::@parse_two
// CHECK:      bmv2ir.transition type  hexstr value #int1_b8i mask #int2_b8i next_state @prs::@parse_two
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_bottom
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:    }
    p4hir.state @parse_one {
      %prs1_one = bmv2ir.symbol_ref @prs1_one : !p4hir.ref<!header_one>
      p4corelib.extract_header %prs_e_0 : <!header_one> from %arg0 : !p4corelib.packet_in
      %val = p4hir.read %prs_e_0 : <!header_one>
      p4hir.assign %val, %prs1_one : <!header_one>
      p4hir.transition to @prs::@parse_two
    }
// CHECK:    bmv2ir.state @parse_one
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_two
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular @prs_e_0
// CHECK:      bmv2ir.assign_header @prs_e_0 to @prs1_one
// CHECK:    }
    p4hir.state @parse_two {
      %prs1_two = bmv2ir.symbol_ref @prs1_two : !p4hir.ref<!header_two>
      p4corelib.extract_header %prs1_two : <!header_two> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_bottom
    }
// CHECK:    bmv2ir.state @parse_two
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_bottom
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular @prs1_two
// CHECK:    }
    p4hir.state @parse_bottom {
      p4hir.transition to @prs::@accept
    }
// CHECK:    bmv2ir.state @parse_bottom
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@accept
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
// CHECK:    bmv2ir.state @accept
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:    }
    p4hir.transition to @prs::@start
  }
}

// -----

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!bit_only = !p4hir.struct<"bit_only", bit: !b8i>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
module {
  bmv2ir.header_instance @prs_only_bit_top_0 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit2 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit1 : !p4hir.ref<!bit_only>
// CHECK:  bmv2ir.header_instance @prs_only_bit_top_0 : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
// CHECK:  bmv2ir.header_instance @prs_only_bit2 : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
// CHECK:  bmv2ir.header_instance @prs_only_bit1 : !bmv2ir.header<"bit_only", [bit:!p4hir.bit<8>], max_length = 1>
  p4hir.parser @prs_only_bit(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!bit_only> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"}, %arg2: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %prs_only_bit_top_0 = bmv2ir.symbol_ref @prs_only_bit_top_0 : !p4hir.ref<!header_top>
    p4hir.state @start {
      %prs_only_bit2 = bmv2ir.symbol_ref @prs_only_bit2 : !p4hir.ref<!header_top>
      p4corelib.extract_header %prs_only_bit_top_0 : <!header_top> from %arg0 : !p4corelib.packet_in
      %skip_field_ref = p4hir.struct_field_ref %prs_only_bit_top_0["skip"] : <!header_top>
      %val = p4hir.read %skip_field_ref : <!b8i>
      %prs_only_bit1 = bmv2ir.symbol_ref @prs_only_bit1 : !p4hir.ref<!bit_only>
      %bit_field_ref = p4hir.struct_field_ref %prs_only_bit1["bit"] : <!bit_only>
      p4hir.assign %val, %bit_field_ref : <!b8i>
      %val_0 = p4hir.read %bit_field_ref : <!b8i>
      %skip_field_ref_1 = p4hir.struct_field_ref %prs_only_bit2["skip"] : <!header_top>
      p4hir.assign %val_0, %skip_field_ref_1 : <!b8i>
      p4hir.transition to @prs_only_bit::@accept
// CHECK:      bmv2ir.extract  regular @prs_only_bit_top_0
// CHECK:      %[[SRC:.*]] = bmv2ir.field @prs_only_bit_top_0["skip"] -> !b8i
// CHECK:      %[[DST:.*]] = bmv2ir.field @prs_only_bit1["bit"] -> !b8i
// CHECK:      bmv2ir.assign %[[SRC]] : !b8i to %[[DST]] : !b8i
// CHECK:      %[[DST2:.*]] = bmv2ir.field @prs_only_bit2["skip"] -> !b8i
// CHECK:      bmv2ir.assign %[[DST]] : !b8i to %[[DST2]] : !b8i
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_only_bit::@start
  }
}

// -----


!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!bit_only = !p4hir.struct<"bit_only", bit: !b8i>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
module {
  bmv2ir.header_instance @prs_only_bit_top_0 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit2 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit1 : !p4hir.ref<!bit_only>
  p4hir.parser @prs_valid(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!bit_only> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"}, %arg2: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %prs_only_bit_top_0 = bmv2ir.symbol_ref @prs_only_bit_top_0 : !p4hir.ref<!header_top>
    p4hir.state @start {
      %prs_only_bit2 = bmv2ir.symbol_ref @prs_only_bit2 : !p4hir.ref<!header_top>
      p4corelib.extract_header %prs_only_bit_top_0 : <!header_top> from %arg0 : !p4corelib.packet_in
      %skip_field_ref = p4hir.struct_field_ref %prs_only_bit_top_0["__valid"] : <!header_top>
// CHECK: bmv2ir.field @prs_only_bit_top_0["$valid$"] -> !b1i
      p4hir.transition to @prs_valid::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_valid::@start
  }
}

// -----

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!bit_only = !p4hir.struct<"bit_only", bit: !b8i>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
#int255_b8i = #p4hir.int<255> : !b8i
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
module {
  bmv2ir.header_instance @prs_only_bit_top_0 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit2 : !p4hir.ref<!header_top>
  bmv2ir.header_instance @prs_only_bit1 : !p4hir.ref<!bit_only>
  p4hir.parser @p(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!bit_only> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"}, %arg2: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
// CHECK:    bmv2ir.state @start
// CHECK:     transition_key {
// CHECK:      %0 = bmv2ir.field @prs_only_bit2["skip"] -> !b8i
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  hexstr value #int-1_b8i next_state @p::@parse_h
// CHECK:      bmv2ir.transition type  default next_state @p::@accept
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular @prs_only_bit2
// CHECK:    }
    p4hir.state @start {
      %prs_only_bit2 = bmv2ir.symbol_ref @prs_only_bit2 : !p4hir.ref<!header_top>
      p4corelib.extract_header %prs_only_bit2 : <!header_top> from %arg0 : !p4corelib.packet_in
      %r = p4hir.read %prs_only_bit2 : <!header_top>
      %f_ref = p4hir.struct_extract %r["skip"] : !header_top
      p4hir.transition_select %f_ref : !b8i {
        p4hir.select_case {
          %c255_b8i = p4hir.const #int255_b8i
          %set = p4hir.set (%c255_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @p::@parse_h
        p4hir.select_case {
          %everything = p4hir.const #everything
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p::@accept
      }
    }
    p4hir.state @parse_h {
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
}

// -----


!b16i = !p4hir.bit<16>
!b19i = !p4hir.bit<19>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b3i = !p4hir.bit<3>
!b48i = !p4hir.bit<48>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!dummy_apply_res = !p4hir.struct<"dummy_apply_res">
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!metadata = !p4hir.struct<"metadata">
!packet_in = !p4hir.extern<"packet_in">
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
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!H = !p4hir.header<"H", s: !b8i, v: !p4hir.varbit<32>, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, priority: !b3i {alias = ["intrinsic_metadata.priority"]}, _padding: !b3i>
!standard_metadata_t1 = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int0_b9i = #p4hir.int<0> : !b9i
#int1024_infint = #p4hir.int<1024> : !infint
#int1_b9i = #p4hir.int<1> : !b9i
#int32_b32i = #p4hir.int<32> : !b32i
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!headers = !p4hir.struct<"headers", h: !H>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t1>)>
// CHECK: #[[ATTR:.*]] = #p4hir.int<32> : !b32i
module {
  bmv2ir.header_instance @h_0_var_0 : !p4hir.ref<!H>
  bmv2ir.header_instance @standard_metadata_t : !p4hir.ref<!standard_metadata_t>
  bmv2ir.header_instance @headers_h : !p4hir.ref<!H>
  p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "p"}, "vr" : !VerifyChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "vr"}, "ig" : !Ingress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ig"}, "eg" : !Egress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "eg"}, "ck" : !ComputeChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ck"}, "dep" : !Deparser_type_H {p4hir.dir = #undir, p4hir.param_name = "dep"})
  p4hir.parser @parser(%arg0: !p4corelib.packet_in {p4hir.dir = #undir, p4hir.param_name = "b"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg3: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "stdmeta"})() {
    %c32_b32i = p4hir.const #int32_b32i
// CHECK:  %[[CONST:.*]] = p4hir.const #[[ATTR]]
// CHECK:    bmv2ir.state @start
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @parser::@accept
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract_vl  regular @headers_h(%[[CONST]] : !b32i)
// CHECK:    }
    p4hir.state @start {
      p4hir.scope {
        %headers_h = bmv2ir.symbol_ref @headers_h : !p4hir.ref<!H>
        p4corelib.extract_header_variable %headers_h<%c32_b32i> : <!H> from %arg0 : !p4corelib.packet_in
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
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "stdmeta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t1> {p4hir.dir = #inout, p4hir.param_name = "stdmeta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @vc(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @uc(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_apply {
    }
  }
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !headers {p4hir.dir = #in, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      %headers_h = bmv2ir.symbol_ref @headers_h : !p4hir.ref<!H>
      %val = p4hir.read %headers_h : <!H>
      p4corelib.emit %val : !H to %arg0 : !p4corelib.packet_out
    }
  }
  bmv2ir.v1switch @main parser @parser, verify_checksum @vc, ingress @ingress, egress @egress, compute_checksum @uc, deparser @deparser
}
