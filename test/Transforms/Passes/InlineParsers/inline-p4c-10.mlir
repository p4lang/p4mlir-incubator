// RUN: p4mlir-opt --p4hir-simplify-parsers --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Ported from p4c/testdata/p4_16_samples/parser-inline/
// Just test it compiles
// CHECK-LABEL: module

!CloneType = !p4hir.enum<"CloneType", I2E, E2E>
!CounterType = !p4hir.enum<"CounterType", packets, bytes, packets_and_bytes>
!HashAlgorithm = !p4hir.enum<"HashAlgorithm", crc32, crc32_custom, crc16, crc16_custom, random, identity, csum16, xor16>
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
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#in = #p4hir<dir in>
#inout = #p4hir<dir inout>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H> annotations {deparser = []}, (!packet_out, !type_H)>
!data_t = !p4hir.header<"data_t", f: !b8i, __valid: !validity_bit>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t" {metadata = [], name = "standard_metadata"}, ingress_port: !b9i, egress_spec: !b9i, egress_port: !b9i, instance_type: !b32i, packet_length: !b32i, enq_timestamp: !b32i {alias = ["queueing_metadata.enq_timestamp"]}, enq_qdepth: !b19i {alias = ["queueing_metadata.enq_qdepth"]}, deq_timedelta: !b32i {alias = ["queueing_metadata.deq_timedelta"]}, deq_qdepth: !b19i {alias = ["queueing_metadata.deq_qdepth"]}, ingress_global_timestamp: !b48i {alias = ["intrinsic_metadata.ingress_global_timestamp"]}, egress_global_timestamp: !b48i {alias = ["intrinsic_metadata.egress_global_timestamp"]}, mcast_grp: !b16i {alias = ["intrinsic_metadata.mcast_grp"]}, egress_rid: !b16i {alias = ["intrinsic_metadata.egress_rid"]}, checksum_error: !b1i, parser_error: !error, priority: !b3i {alias = ["intrinsic_metadata.priority"]}>
#int0_b9i = #p4hir.int<0> : !b9i
#int10_b9i = #p4hir.int<10> : !b9i
#int1_b9i = #p4hir.int<1> : !b9i
#int20180101_b32i = #p4hir.int<20180101> : !b32i
#int2_b9i = #p4hir.int<2> : !b9i
#int42_b8i = #p4hir.int<42> : !b8i
#int42_infint = #p4hir.int<42> : !infint
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!headers = !p4hir.struct<"headers", h1: !data_t, h2: !data_t, h3: !data_t>
!DeparserImpl = !p4hir.control<"DeparserImpl", (!packet_out, !headers)>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M> annotations {pipeline = []}, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!ParserImpl = !p4hir.parser<"ParserImpl", (!packet_in, !p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
!computeChecksum = !p4hir.control<"computeChecksum", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>)>
!egress = !p4hir.control<"egress", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
!ingress = !p4hir.control<"ingress", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>, !p4hir.ref<!standard_metadata_t>)>
!verifyChecksum = !p4hir.control<"verifyChecksum", (!p4hir.ref<!headers>, !p4hir.ref<!metadata>)>
module {
  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @extract_1<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }
  p4hir.extern @packet_out {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  p4hir.func @verify(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, !error {p4hir.dir = #in, p4hir.param_name = "toSignal"})
  p4hir.func action @NoAction() annotations {noWarn = "unused"} {
    p4hir.return
  }
  p4hir.overload_set @static_assert {
    p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool
    p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool
  }
  %__v1model_version = p4hir.const ["__v1model_version"] #int20180101_b32i
  p4hir.extern @counter {
    p4hir.func @counter(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"}, !CounterType {p4hir.dir = #undir, p4hir.param_name = "type"})
    p4hir.func @count(!b32i {p4hir.dir = #in, p4hir.param_name = "index"})
  }
  p4hir.extern @direct_counter {
    p4hir.func @direct_counter(!CounterType {p4hir.dir = #undir, p4hir.param_name = "type"})
    p4hir.func @count()
  }
  p4hir.extern @meter {
    p4hir.func @meter(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"}, !MeterType {p4hir.dir = #undir, p4hir.param_name = "type"})
    p4hir.func @execute_meter<!type_T>(!b32i {p4hir.dir = #in, p4hir.param_name = "index"}, !p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "result"})
  }
  p4hir.extern @direct_meter<[!type_T]> {
    p4hir.func @direct_meter(!MeterType {p4hir.dir = #undir, p4hir.param_name = "type"})
    p4hir.func @read(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "result"})
  }
  p4hir.extern @register<[!type_T]> {
    p4hir.func @register(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"})
    p4hir.func @read(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "result"}, !b32i {p4hir.dir = #in, p4hir.param_name = "index"}) annotations {noSideEffects}
    p4hir.func @write(!b32i {p4hir.dir = #in, p4hir.param_name = "index"}, !type_T {p4hir.dir = #in, p4hir.param_name = "value"})
  }
  p4hir.extern @action_profile {
    p4hir.func @action_profile(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"})
  }
  p4hir.func @random<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "result"}, !type_T {p4hir.dir = #in, p4hir.param_name = "lo"}, !type_T {p4hir.dir = #in, p4hir.param_name = "hi"})
  p4hir.func @digest<!type_T>(!b32i {p4hir.dir = #in, p4hir.param_name = "receiver"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"})
  p4hir.overload_set @mark_to_drop {
    p4hir.func @mark_to_drop_0() annotations {deprecated = "Please use mark_to_drop(standard_metadata) instead."}
    p4hir.func @mark_to_drop_1(!p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "standard_metadata"}) annotations {pure}
  }
  p4hir.func @hash<!type_O, !type_T, !type_D, !type_M>(!p4hir.ref<!type_O> {p4hir.dir = #out, p4hir.param_name = "result"}, !HashAlgorithm {p4hir.dir = #in, p4hir.param_name = "algo"}, !type_T {p4hir.dir = #in, p4hir.param_name = "base"}, !type_D {p4hir.dir = #in, p4hir.param_name = "data"}, !type_M {p4hir.dir = #in, p4hir.param_name = "max"}) annotations {pure}
  p4hir.extern @action_selector {
    p4hir.func @action_selector(!HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algorithm"}, !b32i {p4hir.dir = #undir, p4hir.param_name = "size"}, !b32i {p4hir.dir = #undir, p4hir.param_name = "outputWidth"})
  }
  p4hir.extern @Checksum16 annotations {deprecated = "Please use verify_checksum/update_checksum instead."} {
    p4hir.func @Checksum16()
    p4hir.func @get<!type_D>(!type_D {p4hir.dir = #in, p4hir.param_name = "data"}) -> !b16i
  }
  p4hir.func @verify_checksum<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !type_O {p4hir.dir = #in, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"})
  p4hir.func @update_checksum<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !p4hir.ref<!type_O> {p4hir.dir = #inout, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"}) annotations {pure}
  p4hir.func @verify_checksum_with_payload<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !type_O {p4hir.dir = #in, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"})
  p4hir.func @update_checksum_with_payload<!type_T, !type_O>(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "condition"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}, !p4hir.ref<!type_O> {p4hir.dir = #inout, p4hir.param_name = "checksum"}, !HashAlgorithm {p4hir.dir = #undir, p4hir.param_name = "algo"}) annotations {noSideEffects}
  p4hir.func @clone(!CloneType {p4hir.dir = #in, p4hir.param_name = "type"}, !b32i {p4hir.dir = #in, p4hir.param_name = "session"})
  p4hir.func @resubmit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "data"}) annotations {deprecated = "Please use 'resubmit_preserving_field_list' instead"}
  p4hir.func @resubmit_preserving_field_list(!b8i {p4hir.dir = #undir, p4hir.param_name = "index"})
  p4hir.func @recirculate<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "data"}) annotations {deprecated = "Please use 'recirculate_preserving_field_list' instead"}
  p4hir.func @recirculate_preserving_field_list(!b8i {p4hir.dir = #undir, p4hir.param_name = "index"})
  p4hir.func @clone3<!type_T>(!CloneType {p4hir.dir = #in, p4hir.param_name = "type"}, !b32i {p4hir.dir = #in, p4hir.param_name = "session"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"}) annotations {deprecated = "Please use 'clone_preserving_field_list' instead"}
  p4hir.func @clone_preserving_field_list(!CloneType {p4hir.dir = #in, p4hir.param_name = "type"}, !b32i {p4hir.dir = #in, p4hir.param_name = "session"}, !b8i {p4hir.dir = #undir, p4hir.param_name = "index"})
  p4hir.func @truncate(!b32i {p4hir.dir = #in, p4hir.param_name = "length"})
  p4hir.func @assert(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"})
  p4hir.func @assume(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"})
  p4hir.overload_set @log_msg {
    p4hir.func @log_msg_0(!string {p4hir.dir = #undir, p4hir.param_name = "msg"})
    p4hir.func @log_msg_1<!type_T>(!string {p4hir.dir = #undir, p4hir.param_name = "msg"}, !type_T {p4hir.dir = #in, p4hir.param_name = "data"})
  }
  p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "p"}, "vr" : !VerifyChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "vr"}, "ig" : !Ingress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ig"}, "eg" : !Egress_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "eg"}, "ck" : !ComputeChecksum_type_H_type_M {p4hir.dir = #undir, p4hir.param_name = "ck"}, "dep" : !Deparser_type_H {p4hir.dir = #undir, p4hir.param_name = "dep"})
  p4hir.parser @Subparser(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!data_t> {p4hir.dir = #inout, p4hir.param_name = "hdr"})() {
    p4hir.state @start {
      %f_field_ref = p4hir.struct_extract_ref %arg1["f"] : <!data_t>
      %c42_b8i = p4hir.const #int42_b8i
      %cast = p4hir.cast(%c42_b8i : !b8i) : !b8i
      p4hir.assign %cast, %f_field_ref : <!b8i>
      p4hir.transition to @Subparser::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @Subparser::@start
  }
  p4hir.parser @ParserImpl(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg3: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "standard_metadata"})() {
    p4hir.instantiate @Subparser () as @p
    p4hir.state @start {
      p4hir.scope {
        %h1_field_ref = p4hir.struct_extract_ref %arg1["h1"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!data_t>
        p4hir.call_method @packet_in::@extract<[!data_t]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_out_arg : <!data_t>
        p4hir.assign %val_0, %h1_field_ref : <!data_t>
      }
      p4hir.scope {
        %h2_field_ref = p4hir.struct_extract_ref %arg1["h2"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!data_t>
        p4hir.call_method @packet_in::@extract<[!data_t]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_out_arg : <!data_t>
        p4hir.assign %val_0, %h2_field_ref : <!data_t>
      }
      %val = p4hir.read %arg3 : <!standard_metadata_t>
      %ingress_port = p4hir.struct_extract %val["ingress_port"] : !standard_metadata_t
      p4hir.transition_select %ingress_port : !b9i {
        p4hir.select_case {
          %c0_b9i = p4hir.const #int0_b9i
          %set = p4hir.set (%c0_b9i) : !p4hir.set<!b9i>
          p4hir.yield %set : !p4hir.set<!b9i>
        } to @ParserImpl::@p0
        p4hir.select_case {
          %c1_b9i = p4hir.const #int1_b9i
          %set = p4hir.set (%c1_b9i) : !p4hir.set<!b9i>
          p4hir.yield %set : !p4hir.set<!b9i>
        } to @ParserImpl::@p1
        p4hir.select_case {
          %c2_b9i = p4hir.const #int2_b9i
          %set = p4hir.set (%c2_b9i) : !p4hir.set<!b9i>
          p4hir.yield %set : !p4hir.set<!b9i>
        } to @ParserImpl::@p2
        p4hir.select_case {
          %everything = p4hir.const #everything
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @ParserImpl::@p3
      }
    }
    p4hir.state @p0 {
      p4hir.scope {
        %h1_field_ref = p4hir.struct_extract_ref %arg1["h1"] : <!headers>
        %hdr_inout_arg = p4hir.variable ["hdr_inout_arg", init] : <!data_t>
        %val = p4hir.read %h1_field_ref : <!data_t>
        p4hir.assign %val, %hdr_inout_arg : <!data_t>
        p4hir.apply @ParserImpl::@p(%arg0, %hdr_inout_arg) : (!packet_in, !p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_inout_arg : <!data_t>
        p4hir.assign %val_0, %h1_field_ref : <!data_t>
      }
      p4hir.transition to @ParserImpl::@accept
    }
    p4hir.state @p1 {
      p4hir.scope {
        %h1_field_ref = p4hir.struct_extract_ref %arg1["h1"] : <!headers>
        %hdr_inout_arg = p4hir.variable ["hdr_inout_arg", init] : <!data_t>
        %val = p4hir.read %h1_field_ref : <!data_t>
        p4hir.assign %val, %hdr_inout_arg : <!data_t>
        p4hir.apply @ParserImpl::@p(%arg0, %hdr_inout_arg) : (!packet_in, !p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_inout_arg : <!data_t>
        p4hir.assign %val_0, %h1_field_ref : <!data_t>
      }
      p4hir.transition to @ParserImpl::@accept
    }
    p4hir.state @p2 {
      p4hir.scope {
        %h2_field_ref = p4hir.struct_extract_ref %arg1["h2"] : <!headers>
        %hdr_inout_arg = p4hir.variable ["hdr_inout_arg", init] : <!data_t>
        %val = p4hir.read %h2_field_ref : <!data_t>
        p4hir.assign %val, %hdr_inout_arg : <!data_t>
        p4hir.apply @ParserImpl::@p(%arg0, %hdr_inout_arg) : (!packet_in, !p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_inout_arg : <!data_t>
        p4hir.assign %val_0, %h2_field_ref : <!data_t>
      }
      p4hir.transition to @ParserImpl::@p4
    }
    p4hir.state @p3 {
      p4hir.scope {
        %h2_field_ref = p4hir.struct_extract_ref %arg1["h2"] : <!headers>
        %hdr_inout_arg = p4hir.variable ["hdr_inout_arg", init] : <!data_t>
        %val = p4hir.read %h2_field_ref : <!data_t>
        p4hir.assign %val, %hdr_inout_arg : <!data_t>
        p4hir.apply @ParserImpl::@p(%arg0, %hdr_inout_arg) : (!packet_in, !p4hir.ref<!data_t>) -> ()
        %val_0 = p4hir.read %hdr_inout_arg : <!data_t>
        p4hir.assign %val_0, %h2_field_ref : <!data_t>
      }
      p4hir.transition to @ParserImpl::@accept
    }
    p4hir.state @p4 {
      p4hir.scope {
        %h3_field_ref = p4hir.struct_extract_ref %arg1["h3"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!data_t>
        p4hir.call_method @packet_in::@extract<[!data_t]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!data_t>) -> ()
        %val = p4hir.read %hdr_out_arg : <!data_t>
        p4hir.assign %val, %h3_field_ref : <!data_t>
      }
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
  p4hir.control @ingress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "standard_metadata"})() {
    p4hir.control_local @__local_ingress_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_ingress_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_local @__local_ingress_standard_metadata_0 = %arg2 : !p4hir.ref<!standard_metadata_t>
    p4hir.control_apply {
      %val = p4hir.read %arg0 : <!headers>
      %h1 = p4hir.struct_extract %val["h1"] : !headers
      %f = p4hir.struct_extract %h1["f"] : !data_t
      %c42 = p4hir.const #int42_infint
      %cast = p4hir.cast(%c42 : !infint) : !b8i
      %eq = p4hir.cmp(eq, %f : !b8i, %cast : !b8i)
      p4hir.if %eq {
        %egress_spec_field_ref = p4hir.struct_extract_ref %arg2["egress_spec"] : <!standard_metadata_t>
        %c1_b9i = p4hir.const #int1_b9i
        %cast_0 = p4hir.cast(%c1_b9i : !b9i) : !b9i
        p4hir.assign %cast_0, %egress_spec_field_ref : <!b9i>
      } else {
        %egress_spec_field_ref = p4hir.struct_extract_ref %arg2["egress_spec"] : <!standard_metadata_t>
        %c10_b9i = p4hir.const #int10_b9i
        %cast_0 = p4hir.cast(%c10_b9i : !b9i) : !b9i
        p4hir.assign %cast_0, %egress_spec_field_ref : <!b9i>
      }
    }
  }
  p4hir.control @egress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"}, %arg2: !p4hir.ref<!standard_metadata_t> {p4hir.dir = #inout, p4hir.param_name = "standard_metadata"})() {
    p4hir.control_local @__local_egress_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_egress_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_local @__local_egress_standard_metadata_0 = %arg2 : !p4hir.ref<!standard_metadata_t>
    p4hir.control_apply {
    }
  }
  p4hir.control @DeparserImpl(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !headers {p4hir.dir = #in, p4hir.param_name = "hdr"})() {
    p4hir.control_local @__local_DeparserImpl_packet_0 = %arg0 : !packet_out
    p4hir.control_local @__local_DeparserImpl_hdr_0 = %arg1 : !headers
    p4hir.control_apply {
      %h1 = p4hir.struct_extract %arg1["h1"] : !headers
      p4hir.call_method @packet_out::@emit<[!data_t]> of %arg0 : !packet_out (%h1) : (!data_t) -> ()
      %h2 = p4hir.struct_extract %arg1["h2"] : !headers
      p4hir.call_method @packet_out::@emit<[!data_t]> of %arg0 : !packet_out (%h2) : (!data_t) -> ()
    }
  }
  p4hir.control @verifyChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_local @__local_verifyChecksum_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_verifyChecksum_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_apply {
    }
  }
  p4hir.control @computeChecksum(%arg0: !p4hir.ref<!headers> {p4hir.dir = #inout, p4hir.param_name = "hdr"}, %arg1: !p4hir.ref<!metadata> {p4hir.dir = #inout, p4hir.param_name = "meta"})() {
    p4hir.control_local @__local_computeChecksum_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.control_local @__local_computeChecksum_meta_0 = %arg1 : !p4hir.ref<!metadata>
    p4hir.control_apply {
    }
  }
  %ParserImpl = p4hir.construct @ParserImpl () : !ParserImpl
  %verifyChecksum = p4hir.construct @verifyChecksum () : !verifyChecksum
  %ingress = p4hir.construct @ingress () : !ingress
  %egress = p4hir.construct @egress () : !egress
  %computeChecksum = p4hir.construct @computeChecksum () : !computeChecksum
  %DeparserImpl = p4hir.construct @DeparserImpl () : !DeparserImpl
  p4hir.instantiate @V1Switch<[!headers, !metadata]> (%ParserImpl, %verifyChecksum, %ingress, %egress, %computeChecksum, %DeparserImpl : !ParserImpl, !verifyChecksum, !ingress, !egress, !computeChecksum, !DeparserImpl) as @main
}

