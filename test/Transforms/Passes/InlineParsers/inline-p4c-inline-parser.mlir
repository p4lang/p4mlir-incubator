// RUN: p4mlir-opt --p4hir-simplify-parsers --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Ported from p4c/testdata/p4_16_samples/inline-parser.p4
// Just test it compiles
// CHECK-LABEL: module

!b32i = !p4hir.bit<32>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!packet_in = !p4hir.extern<"packet_in">
!string = !p4hir.string
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!Header = !p4hir.header<"Header", data: !b32i, __valid: !validity_bit>
#int0_infint = #p4hir.int<0> : !infint
#int1_infint = #p4hir.int<1> : !infint
!arr_2xHeader = !p4hir.array<2x!Header>
!hs_2xHeader = !p4hir.header_stack<2x!Header>
!p1_ = !p4hir.parser<"p1", (!packet_in, !p4hir.ref<!hs_2xHeader>)>
!proto = !p4hir.parser<"proto", (!packet_in, !p4hir.ref<!hs_2xHeader>)>
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
  p4hir.parser @p0(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Header> {p4hir.dir = #out, p4hir.param_name = "h"})() {
    p4hir.state @start {
      p4hir.transition to @p0::@next
    }
    p4hir.state @next {
      p4hir.scope {
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!Header>
        p4hir.call_method @packet_in::@extract<[!Header]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!Header>) -> ()
        %val = p4hir.read %hdr_out_arg : <!Header>
        p4hir.assign %val, %arg1 : <!Header>
      }
      p4hir.transition to @p0::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p0::@start
  }
  p4hir.parser @p1(%arg0: !packet_in {p4hir.dir = #undir, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!hs_2xHeader> {p4hir.dir = #out, p4hir.param_name = "h"})() {
    p4hir.instantiate @p0 () as @p0inst
    p4hir.state @start {
      p4hir.scope {
        %c0 = p4hir.const #int0_infint
        %data_field_ref = p4hir.struct_extract_ref %arg1["data"] : <!hs_2xHeader>
        %cast = p4hir.cast(%c0 : !infint) : !b32i
        %elt_ref = p4hir.array_element_ref %data_field_ref[%cast] : !p4hir.ref<!arr_2xHeader>, !b32i
        %h_out_arg = p4hir.variable ["h_out_arg"] : <!Header>
        p4hir.apply @p1::@p0inst(%arg0, %h_out_arg) : (!packet_in, !p4hir.ref<!Header>) -> ()
        %val = p4hir.read %h_out_arg : <!Header>
        p4hir.assign %val, %elt_ref : <!Header>
      }
      p4hir.scope {
        %c1 = p4hir.const #int1_infint
        %data_field_ref = p4hir.struct_extract_ref %arg1["data"] : <!hs_2xHeader>
        %cast = p4hir.cast(%c1 : !infint) : !b32i
        %elt_ref = p4hir.array_element_ref %data_field_ref[%cast] : !p4hir.ref<!arr_2xHeader>, !b32i
        %h_out_arg = p4hir.variable ["h_out_arg"] : <!Header>
        p4hir.apply @p1::@p0inst(%arg0, %h_out_arg) : (!packet_in, !p4hir.ref<!Header>) -> ()
        %val = p4hir.read %h_out_arg : <!Header>
        p4hir.assign %val, %elt_ref : <!Header>
      }
      p4hir.transition to @p1::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p1::@start
  }
  p4hir.package @top("_p" : !proto {p4hir.dir = #undir, p4hir.param_name = "_p"})
  %p1 = p4hir.construct @p1 () : !p1_
  p4hir.instantiate @top (%p1 : !p1_) as @main
}

