// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --pass-pipeline='builtin.module(p4hir.parser(p4hir-make-headers-toplevel, canonicalize))' < %s | FileCheck %s

// Test assignment to field that is non-existent after flattening.

// Based on:
//   header header_t {
//       bit<8> a;
//       bit<32> b;
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
//       header_t h2;
//       meta_t meta2;
//   }
//
//   struct headers_t {
//       wrap_t wrap;
//       bit<4> unrelated;
//   }
//
//   struct metadata_t {
//       bit<4> v;
//   }
//
//   ...
//   hdrs.wrap = pkt.lookahead<wrap_t>();

!b32i = !p4hir.bit<32>
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!packet_in = !p4hir.extern<"packet_in">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
!header_t = !p4hir.header<"header_t", a: !b8i, b: !b32i, __valid: !validity_bit>
!meta_t = !p4hir.struct<"meta_t", m1: !b4i, m2: !b4i>
!metadata_t = !p4hir.struct<"metadata_t", v: !b4i>
!wrap_t = !p4hir.struct<"wrap_t", h1: !header_t, meta1: !b8i, h2: !header_t, meta2: !meta_t>
!headers_t = !p4hir.struct<"headers_t", wrap: !wrap_t, unrelated: !b4i>

// CHECK-DAG: ![[NEW_HDRS_TYPE:.*]] = !p4hir.struct<"headers_t", wrap_h1: !header_t, wrap_h2: !header_t>
// CHECK-DAG: ![[HDR_META_TYPE:.*]] = !p4hir.struct<"headers_t_Metadata", wrap_meta1: !b8i, wrap_meta2: !meta_t, unrelated: !b4i>
// CHECK-DAG: ![[NEW_META_TYPE:.*]] = !p4hir.struct<"metadata_t", v: !b4i, hdr_metadata$: ![[HDR_META_TYPE]]>

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
    p4hir.state @start {
      %wrap_field_ref = p4hir.struct_field_ref %arg0["wrap"] : <!headers_t>
      %0 = p4hir.call_method @p4_main::@packet_in::@lookahead<[!wrap_t]> () of %arg2 : !packet_in : () -> !wrap_t
      p4hir.assign %0, %wrap_field_ref : <!wrap_t>

      // CHECK-DAG: %[[LOOKAHEAD:.*]] = p4hir.call_method @p4_main::@packet_in::@lookahead<[!wrap_t]> () of %arg2 : !packet_in : () -> !wrap_t

      // CHECK-DAG: %[[H1:.*]] = p4hir.struct_extract %[[LOOKAHEAD]]["h1"] : !wrap_t
      // CHECK-DAG: %[[WRAP_H1_REF:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h1"] : <!headers_t>
      // CHECK-DAG: p4hir.assign %[[H1]], %[[WRAP_H1_REF]] : <!header_t>

      // CHECK-DAG: %[[META1:.*]] = p4hir.struct_extract %[[LOOKAHEAD]]["meta1"] : !wrap_t
      // CHECK-DAG: %[[HDR_METADATA_SPECIAL_REF:.*]] = p4hir.struct_field_ref %[[NEW_META]]["hdr_metadata$"] : <!metadata_t>
      // CHECK-DAG: %[[WRAP_META1_REF:.*]] = p4hir.struct_field_ref %[[HDR_METADATA_SPECIAL_REF]]["wrap_meta1"] : <!headers_t_Metadata>
      // CHECK-DAG: p4hir.assign %[[META1]], %[[WRAP_META1_REF]] : <!b8i>

      // CHECK-DAG: %[[H2:.*]] = p4hir.struct_extract %[[LOOKAHEAD]]["h2"] : !wrap_t
      // CHECK-DAG: %[[WRAP_H2_REF:.*]] = p4hir.struct_field_ref %[[NEW_HDRS]]["wrap_h2"] : <!headers_t>
      // CHECK-DAG: p4hir.assign %[[H2]], %[[WRAP_H2_REF]] : <!header_t>

      // CHECK-DAG: %[[META2:.*]] = p4hir.struct_extract %[[LOOKAHEAD]]["meta2"] : !wrap_t
      // CHECK-DAG: %[[HDR_METADATA_SPECIAL_REF_2:.*]] = p4hir.struct_field_ref %[[NEW_META]]["hdr_metadata$"] : <!metadata_t>
      // CHECK-DAG: %[[WRAP_META2_REF:.*]] = p4hir.struct_field_ref %[[HDR_METADATA_SPECIAL_REF_2]]["wrap_meta2"] : <!headers_t_Metadata>
      // CHECK-DAG: p4hir.assign %[[META2]], %[[WRAP_META2_REF]] : <!meta_t>

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
