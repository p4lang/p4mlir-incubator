// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(p4hir.parser(p4hir-make-headers-toplevel, canonicalize))' < %s | FileCheck %s

// Based on:
//   struct nested_t {
//       bit<32> b;
//       bit<32> c;
//   }
//
//   header header_t {
//       bit<8> a;
//       nested_t nested;
//   }
//
//   struct meta_t {
//       bit<4> m1;
//       bit<4> m2;
//   }
//
//   struct wrap_t {
//       header_t h1;
//       bit<8> meta1;
//       header_t h2[2];
//       meta_t meta2;
//   }
//
//   struct headers_t {
//       wrap_t wrap;
//   }
//
//   struct metadata_t {
//       bit<4> v;
//   }
//
//   ...
//   pkt.extract(hdrs.wrap.h1);
//   pkt.advance(1);
//
//   meta.v = 2;
//   pkt.advance(1);
//
//   hdrs.wrap.meta2.m2 = (bit<4>) hdrs.wrap.h1.nested.b;
//   pkt.advance(1);
//
//   pkt.extract(hdrs.wrap.h2[0]);
//   pkt.advance(1);
//
//   hdrs.wrap.meta1 = hdrs.wrap.h2[0].a;
//   pkt.advance(1);

!b32i = !p4hir.bit<32>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!packet_in = !p4hir.extern<"packet_in">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
!meta_t = !p4hir.struct<"meta_t", m1: !b4i, m2: !b4i>
!metadata_t = !p4hir.struct<"metadata_t", v: !b4i>
!nested_t = !p4hir.struct<"nested_t", b: !b32i, c: !b32i>
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b4i = #p4hir.int<2> : !b4i
!header_t = !p4hir.header<"header_t", a: !b8i, nested: !nested_t, __valid: !validity_bit>
!arr_2xheader_t = !p4hir.array<2x!header_t>
!hs_2xheader_t = !p4hir.header_stack<2x!header_t>
!wrap_t = !p4hir.struct<"wrap_t", h1: !header_t, meta1: !b8i, h2: !hs_2xheader_t, meta2: !meta_t>
!headers_t = !p4hir.struct<"headers_t", wrap: !wrap_t>

// CHECK-DAG: ![[HDR_META_TYPE:.*]] = !p4hir.struct<"headers_t_Metadata", wrap_meta1: !b8i, wrap_meta2: !meta_t>
// CHECK-DAG: ![[NEW_META_TYPE:.*]] = !p4hir.struct<"metadata_t", v: !b4i, hdr_metadata$: ![[HDR_META_TYPE]]>
// CHECK-DAG: ![[NEW_HDRS_TYPE:.*]] = !p4hir.struct<"headers_t", wrap_h1: !header_t, wrap_h2: !hs_2xheader_t>

module @p4_main {
  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @"$fv_rt1T_tZA_P3hdr"<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @"$fv_rt1Tu32_tZA_P18variableSizeHeader23variableFieldSizeInBits"<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }

  // CHECK-LABEL: p4hir.parser @Parser
  // CHECK-SAME: %[[NEW_HDRS:.*]]: !p4hir.ref<![[NEW_HDRS_TYPE]]>
  // CHECK-SAME: %[[NEW_META:.*]]: !p4hir.ref<![[NEW_META_TYPE]]>
  p4hir.parser @Parser(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdrs"}, %arg1: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "meta"}, %arg2: !packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"})() {
    %c0_b32i = p4hir.const #int0_b32i
    %c1_b32i = p4hir.const #int1_b32i
    %c2_b4i = p4hir.const #int2_b4i
    p4hir.state @start {
      // CHECK: p4hir.scope
      p4hir.scope {
        // CHECK: %[[WRAP_H1_REF:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h1"] : <!headers_t>
        // CHECK: p4hir.assign %{{.*}}, %[[WRAP_H1_REF]] : <!header_t>

        %wrap_field_ref_4 = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
        %h1_field_ref_5 = p4hir.struct_field_ref %wrap_field_ref_4["h1"] : <!wrap_t>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!header_t>
        p4hir.call_method @p4_main::@packet_in::@extract::@"$fv_rt1T_tZA_P3hdr"<[!header_t]> (%hdr_out_arg) of %arg2 : !packet_in : (!p4hir.ref<!header_t>) -> ()
        %val_6 = p4hir.read %hdr_out_arg : <!header_t>
        p4hir.assign %val_6, %h1_field_ref_5 : <!header_t>
      }

      // CHECK: p4hir.call_method @p4_main::@packet_in::@advance
      p4hir.call_method @p4_main::@packet_in::@advance (%c1_b32i) of %arg2 : !packet_in : (!b32i) -> ()

      // CHECK: %[[META_V_REF:.*]] = p4hir.struct_field_ref %[[NEW_META]]["v"] : <!metadata_t>
      // CHECK: p4hir.assign %c2_b4i, %[[META_V_REF]] : <!b4i>

      %v_field_ref = p4hir.struct_field_ref %arg1["v"] : <!metadata_t>
      p4hir.assign %c2_b4i, %v_field_ref : <!b4i>

      // CHECK: p4hir.call_method @p4_main::@packet_in::@advance
      p4hir.call_method @p4_main::@packet_in::@advance (%c1_b32i) of %arg2 : !packet_in : (!b32i) -> ()

      // CHECK: %[[WRAP_H1_REF_2:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h1"] : <!headers_t>
      // CHECK: %[[WRAP_H1_NESTED_REF:.*]] = p4hir.struct_field_ref %[[WRAP_H1_REF_2]]["nested"] : <!header_t>
      // CHECK: %[[WRAP_H1_NESTED_REF_B_REF:.*]] = p4hir.struct_field_ref %[[WRAP_H1_NESTED_REF]]["b"] : <!nested_t>
      // CHECK: %[[B32_VAL:.*]] = p4hir.read %[[WRAP_H1_NESTED_REF_B_REF]] : <!b32i>
      // CHECK: %[[B4_VAL:.*]] = p4hir.cast(%[[B32_VAL]] : !b32i) : !b4i
      // CHECK: %[[HDR_METADATA_SPECIAL_REF:.*]] = p4hir.struct_field_ref %[[NEW_META]]["hdr_metadata$"] : <!metadata_t>
      // CHECK: %[[HDR_METADATA_WRAP_META2_REF:.*]] = p4hir.struct_field_ref %[[HDR_METADATA_SPECIAL_REF]]["wrap_meta2"] : <!headers_t_Metadata>
      // CHECK: %[[HDR_METADATA_WRAP_META2_M2_REF:.*]] = p4hir.struct_field_ref %[[HDR_METADATA_WRAP_META2_REF]]["m2"] : <!meta_t>
      // CHECK: p4hir.assign %[[B4_VAL]], %[[HDR_METADATA_WRAP_META2_M2_REF]] : <!b4i>

      %wrap_field_ref = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
      %meta2_field_ref = p4hir.struct_field_ref %wrap_field_ref["meta2"] : <!wrap_t>
      %m2_field_ref = p4hir.struct_field_ref %meta2_field_ref["m2"] : <!meta_t>
      %wrap_field_ref_0 = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
      %h1_field_ref = p4hir.struct_field_ref %wrap_field_ref_0["h1"] : <!wrap_t>
      %nested_field_ref = p4hir.struct_field_ref %h1_field_ref["nested"] : <!header_t>
      %b_field_ref = p4hir.struct_field_ref %nested_field_ref["b"] : <!nested_t>
      %val = p4hir.read %b_field_ref : <!b32i>
      %cast = p4hir.cast(%val : !b32i) : !b4i
      p4hir.assign %cast, %m2_field_ref : <!b4i>

      // CHECK: p4hir.call_method @p4_main::@packet_in::@advance
      p4hir.call_method @p4_main::@packet_in::@advance (%c1_b32i) of %arg2 : !packet_in : (!b32i) -> ()

      // CHECK: p4hir.scope
      p4hir.scope {
        // CHECK: %[[WRAP_H2_REF:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h2"] : <!headers_t>
        // CHECK: %[[WRAP_H2_DATA_REF:.*]] = p4hir.struct_field_ref %[[WRAP_H2_REF]]["data"] : <!hs_2xheader_t>
        // CHECK: %[[WRAP_H2_DATA_0_REF:.*]] = p4hir.array_element_ref %[[WRAP_H2_DATA_REF]][%c0_b32i] : !p4hir.ref<!arr_2xheader_t>, !b32i
        // CHECK: p4hir.assign %{{.*}}, %[[WRAP_H2_DATA_0_REF]] : <!header_t>

        %wrap_field_ref_4 = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
        %h2_field_ref_5 = p4hir.struct_field_ref %wrap_field_ref_4["h2"] : <!wrap_t>
        %data_field_ref_6 = p4hir.struct_field_ref %h2_field_ref_5["data"] : <!hs_2xheader_t>
        %elt_ref_7 = p4hir.array_element_ref %data_field_ref_6[%c0_b32i] : !p4hir.ref<!arr_2xheader_t>, !b32i
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!header_t>
        p4hir.call_method @p4_main::@packet_in::@extract::@"$fv_rt1T_tZA_P3hdr"<[!header_t]> (%hdr_out_arg) of %arg2 : !packet_in : (!p4hir.ref<!header_t>) -> ()
        %val_8 = p4hir.read %hdr_out_arg : <!header_t>
        p4hir.assign %val_8, %elt_ref_7 : <!header_t>
      }

      // CHECK: p4hir.call_method @p4_main::@packet_in::@advance
      p4hir.call_method @p4_main::@packet_in::@advance (%c1_b32i) of %arg2 : !packet_in : (!b32i) -> ()

      // CHECK: %[[WRAP_H2_REF_2:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h2"] : <!headers_t>
      // CHECK: %[[WRAP_H2_DATA_REF_2:.*]] = p4hir.struct_field_ref %[[WRAP_H2_REF_2]]["data"] : <!hs_2xheader_t>
      // CHECK: %[[WRAP_H2_DATA_0_REF_2:.*]] = p4hir.array_element_ref %[[WRAP_H2_DATA_REF_2]][%c0_b32i] : !p4hir.ref<!arr_2xheader_t>, !b32i
      // CHECK: %[[A_REF:.*]] = p4hir.struct_field_ref %[[WRAP_H2_DATA_0_REF_2]]["a"] : <!header_t>
      // CHECK: %[[A_VAL:.*]] = p4hir.read %[[A_REF]] : <!b8i>
      // CHECK: %[[HDR_METADATA_SPECIAL_REF_2:.*]] = p4hir.struct_field_ref %[[NEW_META]]["hdr_metadata$"] : <!metadata_t>
      // CHECK: %[[HDR_METADATA_WRAP_META1_REF:.*]] = p4hir.struct_field_ref %[[HDR_METADATA_SPECIAL_REF_2]]["wrap_meta1"] : <!headers_t_Metadata>
      // CHECK: p4hir.assign %[[A_VAL]], %[[HDR_METADATA_WRAP_META1_REF]] : <!b8i>

      %wrap_field_ref_1 = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
      %meta1_field_ref = p4hir.struct_field_ref %wrap_field_ref_1["meta1"] : <!wrap_t>
      %wrap_field_ref_2 = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
      %h2_field_ref = p4hir.struct_field_ref %wrap_field_ref_2["h2"] : <!wrap_t>
      %data_field_ref = p4hir.struct_field_ref %h2_field_ref["data"] : <!hs_2xheader_t>
      %elt_ref = p4hir.array_element_ref %data_field_ref[%c0_b32i] : !p4hir.ref<!arr_2xheader_t>, !b32i
      %a_field_ref = p4hir.struct_field_ref %elt_ref["a"] : <!header_t>
      %val_3 = p4hir.read %a_field_ref : <!b8i>
      p4hir.assign %val_3, %meta1_field_ref : <!b8i>

      // CHECK: p4hir.call_method @p4_main::@packet_in::@advance
      p4hir.call_method @p4_main::@packet_in::@advance (%c1_b32i) of %arg2 : !packet_in : (!b32i) -> ()

      p4hir.transition to @accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @start
  }
}