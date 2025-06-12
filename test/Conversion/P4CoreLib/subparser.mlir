// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument, BadHeaderType>
!packet_in = !p4hir.extern<"packet_in" annotations {corelib}>
!string = !p4hir.string
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!h1_t = !p4hir.header<"h1_t", hdr_type: !b8i, op1: !b8i, next_hdr_type: !b8i, __valid: !validity_bit>
!h2_t = !p4hir.header<"h2_t", hdr_type: !b8i, f1: !b8i, next_hdr_type: !b8i, __valid: !validity_bit>
!h3_t = !p4hir.header<"h3_t", hdr_type: !b8i, data: !b8i, __valid: !validity_bit>
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i
!headers = !p4hir.struct<"headers", h1: !h1_t, h2: !h2_t, h3: !h3_t>
module {
  p4hir.extern @packet_in annotations {corelib} {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @extract_1<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }
  p4hir.extern @packet_out annotations {corelib} {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  p4hir.func @verify(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, !error {p4hir.dir = #in, p4hir.param_name = "toSignal"}) annotations {corelib}
  p4hir.overload_set @static_assert {
    p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool annotations {corelib}
    p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool annotations {corelib}
  }
  // CHECK-LABEL: @subParserImpl
  // CHECK-SAME: !p4corelib.packet_in
  p4hir.parser @subParserImpl(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"}, %arg2: !p4hir.ref<!b8i> {p4hir.dir = #out, p4hir.param_name = "ret_next_hdr_type"})() {
    p4hir.state @start {
      p4hir.scope {
        %h2_field_ref = p4hir.struct_extract_ref %arg1["h2"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!h2_t>
        p4hir.call_method @packet_in::@extract<[!h2_t]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!h2_t>) -> ()
        %val_0 = p4hir.read %hdr_out_arg : <!h2_t>
        p4hir.assign %val_0, %h2_field_ref : <!h2_t>
      }
      %val = p4hir.read %arg1 : <!headers>
      %h2 = p4hir.struct_extract %val["h2"] : !headers
      %next_hdr_type = p4hir.struct_extract %h2["next_hdr_type"] : !h2_t
      p4hir.assign %next_hdr_type, %arg2 : <!b8i>
      p4hir.transition to @subParserImpl::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @subParserImpl::@start
  }
  // CHECK-LABEL: parserI
  // CHECK-SAME: !p4corelib.packet_in
  p4hir.parser @parserI(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!headers> {p4hir.dir = #out, p4hir.param_name = "hdr"})() {
    p4hir.instantiate @subParserImpl() as @subp
    %my_next_hdr_type = p4hir.variable ["my_next_hdr_type"] : <!b8i>
    p4hir.state @start {
      p4hir.scope {
        %h1_field_ref = p4hir.struct_extract_ref %arg1["h1"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!h1_t>
        p4hir.call_method @packet_in::@extract<[!h1_t]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!h1_t>) -> ()
        %val_0 = p4hir.read %hdr_out_arg : <!h1_t>
        p4hir.assign %val_0, %h1_field_ref : <!h1_t>
      }
      %val = p4hir.read %arg1 : <!headers>
      %h1 = p4hir.struct_extract %val["h1"] : !headers
      %next_hdr_type = p4hir.struct_extract %h1["next_hdr_type"] : !h1_t
      p4hir.transition_select %next_hdr_type : !b8i {
        p4hir.select_case {
          %c2_b8i = p4hir.const #int2_b8i
          %set = p4hir.set (%c2_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @parserI::@parse_first_h2
        p4hir.select_case {
          %c3_b8i = p4hir.const #int3_b8i
          %set = p4hir.set (%c3_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @parserI::@parse_h3
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @parserI::@accept
      }
    }
    p4hir.state @parse_first_h2 {
      p4hir.scope {
        %hdr_inout_arg = p4hir.variable ["hdr_inout_arg", init] : <!headers>
        %val_0 = p4hir.read %arg1 : <!headers>
        p4hir.assign %val_0, %hdr_inout_arg : <!headers>
        %ret_next_hdr_type_out_arg = p4hir.variable ["ret_next_hdr_type_out_arg"] : <!b8i>
        p4hir.apply @subp(%arg0, %hdr_inout_arg, %ret_next_hdr_type_out_arg) : (!packet_in, !p4hir.ref<!headers>, !p4hir.ref<!b8i>) -> ()
        %val_1 = p4hir.read %hdr_inout_arg : <!headers>
        p4hir.assign %val_1, %arg1 : <!headers>
        %val_2 = p4hir.read %ret_next_hdr_type_out_arg : <!b8i>
        p4hir.assign %val_2, %my_next_hdr_type : <!b8i>
      }
      %val = p4hir.read %my_next_hdr_type : <!b8i>
      p4hir.transition_select %val : !b8i {
        p4hir.select_case {
          %c3_b8i = p4hir.const #int3_b8i
          %set = p4hir.set (%c3_b8i) : !p4hir.set<!b8i>
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @parserI::@parse_h3
        p4hir.select_case {
          %everything = p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @parserI::@accept
      }
    }
    p4hir.state @parse_h3 {
      p4hir.scope {
        %h3_field_ref = p4hir.struct_extract_ref %arg1["h3"] : <!headers>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!h3_t>
        p4hir.call_method @packet_in::@extract<[!h3_t]> of %arg0: !packet_in (%hdr_out_arg) : (!p4hir.ref<!h3_t>) -> ()
        %val = p4hir.read %hdr_out_arg : <!h3_t>
        p4hir.assign %val, %h3_field_ref : <!h3_t>
      }
      p4hir.transition to @parserI::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @parserI::@start
  }
}
