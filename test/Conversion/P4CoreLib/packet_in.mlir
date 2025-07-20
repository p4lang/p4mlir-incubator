// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
// CHECK-NOT: !p4hir.extern<"packet_in"
!packet_in = !p4hir.extern<"packet_in" annotations {corelib}>
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
!header_bottom = !p4hir.header<"header_bottom", length: !b8i, data: !p4hir.varbit<256>, __valid: !validity_bit>
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
#int1_b8i = #p4hir.int<1> : !b8i
#int256_b32i = #p4hir.int<256> : !b32i
#int2_b8i = #p4hir.int<2> : !b8i
!Headers_t = !p4hir.struct<"Headers_t", top: !header_top, one: !header_one, two: !header_two, bottom: !header_bottom>
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
  // CHECK-LABEL: prs
  // CHECK-SAME: !p4corelib.packet_in
  p4hir.parser @prs(%arg0: !packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #out, p4hir.param_name = "headers"})() {
    p4hir.state @start {
      %top_field_ref = p4hir.struct_extract_ref %arg1["top"] : <!Headers_t>
      // CHECK: p4corelib.extract_header
      p4hir.call_method @packet_in::@extract<[!header_top]> of %arg0 : !packet_in (%top_field_ref) : (!p4hir.ref<!header_top>) -> ()
      %val = p4hir.read %arg1 : <!Headers_t>
      %top = p4hir.struct_extract %val["top"] : !Headers_t
      %skip = p4hir.struct_extract %top["skip"] : !header_top
      %cast = p4hir.cast(%skip : !b8i) : !b32i
      // CHECK: p4corelib.packet_advance
      p4hir.call_method @packet_in::@advance of %arg0 : !packet_in (%cast) : (!b32i) -> ()
      p4hir.transition to @prs::@parse_headers
    }
    p4hir.state @parse_headers {
      // CHECK: p4corelib.packet_lookahead
      %0 = p4hir.call_method @packet_in::@lookahead<[!b8i]> of %arg0 : !packet_in () : () -> !b8i
      p4hir.transition_select %0 : !b8i {
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
          %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @prs::@parse_bottom
      }
    }
    p4hir.state @parse_one {
      %one_field_ref = p4hir.struct_extract_ref %arg1["one"] : <!Headers_t>
      p4hir.call_method @packet_in::@extract<[!header_one]> of %arg0 : !packet_in (%one_field_ref) : (!p4hir.ref<!header_one>) -> ()
      p4hir.transition to @prs::@parse_headers
    }
    p4hir.state @parse_two {
      %two_field_ref = p4hir.struct_extract_ref %arg1["two"] : <!Headers_t>
      p4hir.call_method @packet_in::@extract<[!header_two]> of %arg0 : !packet_in (%two_field_ref) : (!p4hir.ref<!header_two>) -> ()
      p4hir.transition to @prs::@parse_headers
    }
    p4hir.state @parse_bottom {
      // CHECK: p4corelib.packet_lookahead
      %0 = p4hir.call_method @packet_in::@lookahead<[!b8i]> of %arg0 : !packet_in () : () -> !b8i
      // CHECK: p4corelib.packet_length
      %1 = p4hir.call_method @packet_in::@length of %arg0 : !packet_in () : () -> !b32i
      %c256_b32i = p4hir.const #int256_b32i
      %div = p4hir.binop(div, %1, %c256_b32i) : !b32i
      %cast = p4hir.cast(%div : !b32i) : !b8i
      %add = p4hir.binop(add, %0, %cast) : !b8i
      %bottom_field_ref = p4hir.struct_extract_ref %arg1["bottom"] : <!Headers_t>
      %cast_0 = p4hir.cast(%add : !b8i) : !b32i
      // CHECK: p4corelib.extract_header_variable
      p4hir.call_method @packet_in::@extract<[!header_bottom]> of %arg0 : !packet_in (%bottom_field_ref, %cast_0) : (!p4hir.ref<!header_bottom>, !b32i) -> ()
      p4hir.transition to @prs::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @prs::@start
  }
}

