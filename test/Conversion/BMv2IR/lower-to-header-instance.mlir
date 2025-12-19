// RUN: p4mlir-opt -p='builtin.module(lower-to-header-instance)' --split-input-file %s | FileCheck %s
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
!header_and_bit = !p4hir.struct<"header_and_bit", top: !header_top, bit: !b8i>
module {
// CHECK:  bmv2ir.header_instance @prs_e_0 : !p4hir.ref<!header_one>
// CHECK:  bmv2ir.header_instance @prs1_top : !p4hir.ref<!header_top>
// CHECK:  bmv2ir.header_instance @prs1_one : !p4hir.ref<!header_one>
// CHECK:  bmv2ir.header_instance @prs1_two : !p4hir.ref<!header_two>
  p4hir.parser @prs(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!Headers_t> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %e_0 = p4hir.variable ["e_0"] annotations {name = "ParserImpl.e"} : <!header_one>
    // CHECK: %[[E_0:.*]] = bmv2ir.symbol_ref @prs_e_0 : !p4hir.ref<!header_one>
    p4hir.state @start {
      %top_field_ref = p4hir.struct_field_ref %arg1["top"] : <!Headers_t>
      p4corelib.extract_header %top_field_ref : <!header_top> from %arg0 : !p4corelib.packet_in
// CHECK:  %[[REF:.*]] = bmv2ir.symbol_ref @prs1_top : !p4hir.ref<!header_top>
// CHECK: p4corelib.extract_header %[[REF]] : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_headers
    }
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
    p4hir.state @parse_one {
// CHECK: %[[REF2:.*]] = bmv2ir.symbol_ref @prs1_one : !p4hir.ref<!header_one>
      %one_field_ref = p4hir.struct_field_ref %arg1["one"] : <!Headers_t>
      p4corelib.extract_header %e_0 : <!header_one> from %arg0 : !p4corelib.packet_in
// CHECK: p4corelib.extract_header %[[E_0]] : <!header_one> from %arg0 : !p4corelib.packet_in
      %val = p4hir.read %e_0 : <!header_one>
      p4hir.assign %val, %one_field_ref : <!header_one>
// CHECK: p4hir.assign %{{.*}}, %[[REF2]] : <!header_one>
      p4hir.transition to @prs::@parse_two
    }
    p4hir.state @parse_two {
// CHECK: %[[REF3:.*]] = bmv2ir.symbol_ref @prs1_two : !p4hir.ref<!header_two>
      %two_field_ref = p4hir.struct_field_ref %arg1["two"] : <!Headers_t>
      p4corelib.extract_header %two_field_ref : <!header_two> from %arg0 : !p4corelib.packet_in
// CHECK:  p4corelib.extract_header %[[REF3]] : <!header_two> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs::@parse_bottom
    }
    p4hir.state @parse_bottom {
      p4hir.transition to @prs::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs::@start
  }
}

// -----

// Checks that we correctly handle a header parser arg
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
module {
// CHECK: bmv2ir.header_instance @prs_header_arg1 : !p4hir.ref<!header_top>
  p4hir.parser @prs_header_arg(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    p4hir.state @start {
// CHECK:      %[[REF:.*]] = bmv2ir.symbol_ref @prs_header_arg1 : !p4hir.ref<!header_top>
// CHECK:      p4corelib.extract_header %[[REF]] : <!header_top> from %arg0 : !p4corelib.packet_in
      p4corelib.extract_header %arg1 : <!header_top> from %arg0 : !p4corelib.packet_in
      p4hir.transition to @prs_header_arg::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_header_arg::@start
  }

}

// -----

// Checks that we correctly split a struct having both headers and bits
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!header_and_bit = !p4hir.struct<"header_and_bit", top: !header_top, bit: !b8i>
// CHECK: ![[SPLIT_STRUCT:.*]] = !p4hir.struct<"header_and_bit", bit: !b8i>
module {
// CHECK: bmv2ir.header_instance @prs_header_and_bit1 : !p4hir.ref<![[SPLIT_STRUCT]]>
    p4hir.parser @prs_header_and_bit(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!header_and_bit> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %var = p4hir.variable ["top_0"] annotations {name = "ParserImpl.e"} : <!header_top>
    p4hir.state @start {
      p4corelib.extract_header %var : <!header_top> from %arg0 : !p4corelib.packet_in
      %bit = p4hir.struct_field_ref %var["skip"] : <!header_top>
      %val = p4hir.read %bit : <!b8i>
      %ref = p4hir.struct_field_ref %arg1["bit"] : <!header_and_bit>
// CHECK: %[[REF:.*]] = bmv2ir.symbol_ref @prs_header_and_bit1 : !p4hir.ref<!header_and_bit>
// CHECK: %{{.*}} = p4hir.struct_field_ref %[[REF]]["bit"] : <!header_and_bit>
      p4hir.assign %val, %ref : <!b8i>
      p4hir.transition to @prs_header_and_bit::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_header_and_bit::@start
  }
}

// -----

// Checks that we correctly insert header instances for structs with only bit fields
!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!header_top = !p4hir.header<"header_top", skip: !b8i, __valid: !validity_bit>
!bit_only = !p4hir.struct<"bit_only", bit: !b8i>
module {
// CHECK: bmv2ir.header_instance @prs_only_bit1 : !p4hir.ref<!bit_only>
    p4hir.parser @prs_only_bit(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "p"}, %arg1: !p4hir.ref<!bit_only> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"}, %arg2: !p4hir.ref<!header_top> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "headers"})() {
    %var = p4hir.variable ["top_0"] annotations {name = "ParserImpl.e"} : <!header_top>
    p4hir.state @start {
      p4corelib.extract_header %var : <!header_top> from %arg0 : !p4corelib.packet_in
      %bit = p4hir.struct_field_ref %var["skip"] : <!header_top>
      %val = p4hir.read %bit : <!b8i>
      %ref = p4hir.struct_field_ref %arg1["bit"] : <!bit_only>
      p4hir.assign %val, %ref : <!b8i>
      %val2 = p4hir.read %ref : <!b8i>
      %ref2 = p4hir.struct_field_ref %arg2["skip"] : <!header_top>
      p4hir.assign %val2, %ref2 : <!b8i>
      p4hir.transition to @prs_only_bit::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @prs_only_bit::@start
  }
}
