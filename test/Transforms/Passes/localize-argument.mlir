// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --p4hir-localize-argument < %s | FileCheck %s

// Based on:
//   header header_t {
//       bit<8> a;
//   }
//
//   struct headers_t {
//       header_t h1;
//       header_t h2;
//       bit<8> meta1;
//       bit<8> meta2;
//   }
//
//   parser Parser(inout headers_t hdrs, packet_in pkt) {
//       state start {
//           pkt.extract(hdrs.h1);
//           hdrs.meta1 = 1;
//           transition accept;
//       }
//   }
//
//   control Control(inout headers_t hdrs) {
//       action dummy() {
//           hdrs.h2.a = 1;
//       }
//       apply {
//           hdrs.meta2 = hdrs.h1.a;
//       }
//   }


!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!packet_in = !p4hir.extern<"packet_in">
!type_T = !p4hir.type_var<"T">
!validity_bit = !p4hir.validity.bit
#in = #p4hir<dir in>
#out = #p4hir<dir out>
!header_t = !p4hir.header<"header_t", a: !b8i, __valid: !validity_bit>
#int1_b8i = #p4hir.int<1> : !b8i
!headers_t = !p4hir.struct<"headers_t", h1: !header_t, h2: !header_t, meta1: !b8i, meta2: !b8i>

// CHECK-DAG: ![[PARSER_LOCAL_STATE_TYPE:.*]] = !p4hir.struct<"local_state", meta1: !b8i>
// CHECK-DAG: ![[CONTROL_LOCAL_STATE_TYPE:.*]] = !p4hir.struct<"local_state", h2_a: !b8i, meta2: !b8i>
// CHECK-DAG: ![[SHARED_STATE_TYPE:.*]] = !p4hir.struct<"shared_state", h1: !header_t>

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
  // CHECK-SAME: %arg0: !p4hir.ref<![[SHARED_STATE_TYPE]]>
  p4hir.parser @Parser(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdrs"}, %arg1: !packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"})() {
    // CHECK-DAG: %[[PARSER_LOCAL_STATE:.*]] = p4hir.variable ["local_state$"] : <![[PARSER_LOCAL_STATE_TYPE]]>

    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      p4hir.scope {
        %h1_field_ref = p4hir.struct_field_ref %arg0["h1"] : <!headers_t>
        %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!header_t>
        p4hir.call_method @p4_main::@packet_in::@extract::@"$fv_rt1T_tZA_P3hdr"<[!header_t]> (%hdr_out_arg) of %arg1 : !packet_in : (!p4hir.ref<!header_t>) -> ()
        %val = p4hir.read %hdr_out_arg : <!header_t>
        p4hir.assign %val, %h1_field_ref : <!header_t>

        // CHECK-DAG: %[[H1_REF:.*]] = p4hir.struct_field_ref %arg0["h1"] : <!shared_state>
        // CHECK-DAG: p4hir.assign %{{.*}}, %[[H1_REF]] : <!header_t>
      }

      // CHECK-DAG: %[[LOCAL_META_1:.*]] = p4hir.struct_field_ref %[[PARSER_LOCAL_STATE]]["meta1"] : <![[PARSER_LOCAL_STATE_TYPE]]>
      // CHECK-DAG: p4hir.assign %c1_b8i, %[[LOCAL_META_1]] : <!b8i>

      %meta1_field_ref = p4hir.struct_field_ref %arg0["meta1"] : <!headers_t>
      p4hir.assign %c1_b8i, %meta1_field_ref : <!b8i>
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

  // CHECK-LABEL: p4hir.control @Control
  // CHECK-SAME: %arg0: !p4hir.ref<![[SHARED_STATE_TYPE]]>
  p4hir.control @Control(%arg0: !p4hir.ref<!headers_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdrs"})() {
    // CHECK-DAG: %[[CONTROL_LOCAL_STATE:.*]] = p4hir.variable ["local_state$"] : <![[CONTROL_LOCAL_STATE_TYPE]]>
    // CHECK-DAG: p4hir.control_local @[[CONTROL_LOCAL_STATE_SYM:.*]] = %[[CONTROL_LOCAL_STATE]] : !p4hir.ref<![[CONTROL_LOCAL_STATE_TYPE]]>

    p4hir.control_local @Control_hdrs$ = %arg0 : !p4hir.ref<!headers_t>

    // CHECK-LABEL: p4hir.func action @dummy
    p4hir.func action @dummy() {
      // CHECK-DAG: %[[SYM_REF:.*]] = p4hir.symbol_ref @[[CONTROL_LOCAL_STATE_SYM]] : !p4hir.ref<![[CONTROL_LOCAL_STATE_TYPE]]>
      // CHECK-DAG: %[[LOCAL_H2_A_REF:.*]] = p4hir.struct_field_ref %[[SYM_REF]]["h2_a"] : <![[CONTROL_LOCAL_STATE_TYPE]]>
      // CHECK-DAG: p4hir.assign %c1_b8i, %[[LOCAL_H2_A_REF]] : <!b8i>

      %c1_b8i = p4hir.const #int1_b8i
      %Control_hdrs$ = p4hir.symbol_ref @Control_hdrs$ : !p4hir.ref<!headers_t>
      %h2_field_ref = p4hir.struct_field_ref %Control_hdrs$["h2"] : <!headers_t>
      %a_field_ref = p4hir.struct_field_ref %h2_field_ref["a"] : <!header_t>
      p4hir.assign %c1_b8i, %a_field_ref : <!b8i>
      p4hir.return
    }

    // CHECK-LABEL: p4hir.control_apply
    p4hir.control_apply {
      // CHECK-DAG: %[[H1_REF:.*]] = p4hir.struct_field_ref %arg0["h1"] : <!shared_state>
      // CHECK-DAG: %[[A_REF:.*]] = p4hir.struct_field_ref %[[H1_REF]]["a"] : <!header_t>
      // CHECK-DAG: %[[VAL:.*]] = p4hir.read %[[A_REF]] : <!b8i>
      // CHECK-DAG: %[[LOCAL_META_2:.*]] = p4hir.struct_field_ref %[[CONTROL_LOCAL_STATE]]["meta2"] : <![[CONTROL_LOCAL_STATE_TYPE]]>
      // CHECK-DAG: p4hir.assign %[[VAL]], %[[LOCAL_META_2]] : <!b8i>

      %meta2_field_ref = p4hir.struct_field_ref %arg0["meta2"] : <!headers_t>
      %h1_field_ref = p4hir.struct_field_ref %arg0["h1"] : <!headers_t>
      %a_field_ref = p4hir.struct_field_ref %h1_field_ref["a"] : <!header_t>
      %val = p4hir.read %a_field_ref : <!b8i>
      p4hir.assign %val, %meta2_field_ref : <!b8i>
    }
  }
}
