// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s

!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b48i = !p4hir.bit<48>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!packet_out = !p4hir.extern<"packet_out" annotations {corelib}>
!string = !p4hir.string
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!ethernet_t = !p4hir.header<"ethernet_t", dst_addr: !b48i, src_addr: !b48i, ether_type: !b16i, __valid: !validity_bit>
!headers_t = !p4hir.struct<"headers_t", u0_ethernet: !ethernet_t>
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
  p4hir.func action @NoAction() annotations {noWarn = "unused"} {
    p4hir.return
  }
  p4hir.overload_set @static_assert {
    p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool annotations {corelib}
    p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool annotations {corelib}
  }
  // CHECK-LABEL: p4hir.control
  p4hir.control @deparser(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !headers_t {p4hir.dir = #in, p4hir.param_name = "hdr"})() {
    // CHECK: p4hir.control_local @__local_deparser_packet_0 = %arg0 : !p4corelib.packet_out
    p4hir.control_local @__local_deparser_packet_0 = %arg0 : !packet_out
    p4hir.control_local @__local_deparser_hdr_0 = %arg1 : !headers_t
    p4hir.control_apply {
      %u0_ethernet = p4hir.struct_extract %arg1["u0_ethernet"] : !headers_t
      p4hir.call_method @packet_out::@emit<[!ethernet_t]> (%u0_ethernet) of %arg0 : !packet_out : (!ethernet_t) -> ()
    }
  }
}
