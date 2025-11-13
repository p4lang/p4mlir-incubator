// RUN: p4mlir-opt -p='builtin.module(p4hir-to-bmv2ir)' %s | FileCheck %s
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
!header_bottom = !p4hir.header<"header_bottom", length: !b8i, data: !p4hir.varbit<256>, __valid: !validity_bit>
!header_one = !p4hir.header<"header_one", type: !b8i, data: !b8i, __valid: !validity_bit>
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_two = !p4hir.header<"header_two", type: !b8i, data: !b16i, __valid: !validity_bit>
#int1_b8i = #p4hir.int<1> : !b8i
#int256_b32i = #p4hir.int<256> : !b32i
#int2_b8i = #p4hir.int<2> : !b8i
!Headers_t = !p4hir.struct<"Headers_t", top: !header_top, one: !header_one, two: !header_two, bottom: !header_bottom>
module {
  p4hir.parser @prs(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
// CHECK:  bmv2ir.parser @prs init_state @prs::@start {
    p4hir.state @start {
      %top_field_ref = p4hir.struct_field_ref %arg1["top"] : <!Headers_t>
      p4corelib.extract_header %top_field_ref : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_headers
    }
// CHECK:    bmv2ir.state @start
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_headers
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular "top"
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
      %one_field_ref = p4hir.struct_field_ref %arg1["one"] : <!Headers_t>
      p4corelib.extract_header %one_field_ref : <!header_one> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_two
    }
// CHECK:    bmv2ir.state @parse_one
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_two
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular "one"
// CHECK:    }
    p4hir.state @parse_two {
      %two_field_ref = p4hir.struct_field_ref %arg1["two"] : <!Headers_t>
      p4corelib.extract_header %two_field_ref : <!header_two> from %arg0 : !p4corelib.packet_in
      //p4hir.transition to @prs::@parse_headers
      p4hir.transition to @prs::@parse_bottom
    }
// CHECK:    bmv2ir.state @parse_two
// CHECK:     transition_key {
// CHECK:    }
// CHECK:     transitions {
// CHECK:      bmv2ir.transition type  default next_state @prs::@parse_bottom
// CHECK:    }
// CHECK:     parser_ops {
// CHECK:      bmv2ir.extract  regular "two"
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
