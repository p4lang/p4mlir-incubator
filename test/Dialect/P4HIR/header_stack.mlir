// RUN: p4mlir-opt %s --verify-roundtrip | FileCheck %s

!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", __valid: !validity_bit>
!hh = !p4hir.header<"hh", __valid: !validity_bit>
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int4_b32i = #p4hir.int<4> : !b32i
#invalid = #p4hir<validity.bit invalid> : !validity_bit
#valid = #p4hir<validity.bit valid> : !validity_bit
!arr_4xh = !p4hir.array<4x!h>
!hs_4xh = !p4hir.header_stack<4x!h>
!hu = !p4hir.header_union<"hu", h1: !h, h2: !hh>
!arr_2xhu = !p4hir.array<2x!hu>
!hs_2xhu = !p4hir.header_stack<2x!hu>
// CHECK: module
module {
  p4hir.parser @p()() {
    p4hir.state @start {
      %stack = p4hir.variable ["stack"] : <!hs_4xh>
      %c3_b32i = p4hir.const #int3_b32i
      %data_field_ref = p4hir.struct_extract_ref %stack["data"] : <!hs_4xh>
      %elt_ref = p4hir.array_element_ref %data_field_ref[%c3_b32i] : !p4hir.ref<!arr_4xh>, !b32i
      %valid = p4hir.const #valid
      %__valid_field_ref = p4hir.struct_extract_ref %elt_ref["__valid"] : <!h>
      p4hir.assign %valid, %__valid_field_ref : <!validity_bit>
      %c3_b32i_0 = p4hir.const #int3_b32i
      %val = p4hir.read %stack : <!hs_4xh>
      %data = p4hir.struct_extract %val["data"] : !hs_4xh
      %array_elt = p4hir.array_get %data[%c3_b32i_0] : !arr_4xh, !b32i
      %b = p4hir.variable ["b", init] : <!h>
      p4hir.assign %array_elt, %b : <!h>
      %val_1 = p4hir.read %stack : <!hs_4xh>
      %data_2 = p4hir.struct_extract %val_1["data"] : !hs_4xh
      %nextIndex = p4hir.struct_extract %val_1["nextIndex"] : !hs_4xh
      %c1_b32i = p4hir.const #int1_b32i
      %sub = p4hir.binop(sub, %nextIndex, %c1_b32i) : !b32i
      %array_elt_3 = p4hir.array_get %data_2[%sub] : !arr_4xh, !b32i
      p4hir.assign %array_elt_3, %b : <!h>
      %c3_b32i_4 = p4hir.const #int3_b32i
      %data_field_ref_5 = p4hir.struct_extract_ref %stack["data"] : <!hs_4xh>
      %elt_ref_6 = p4hir.array_element_ref %data_field_ref_5[%c3_b32i_4] : !p4hir.ref<!arr_4xh>, !b32i
      %val_7 = p4hir.read %b : <!h>
      p4hir.assign %val_7, %elt_ref_6 : <!h>
      %data_field_ref_8 = p4hir.struct_extract_ref %stack["data"] : <!hs_4xh>
      %nextIndex_field_ref = p4hir.struct_extract_ref %stack["nextIndex"] : <!hs_4xh>
      %val_9 = p4hir.read %nextIndex_field_ref : <!b32i>
      %elt_ref_10 = p4hir.array_element_ref %data_field_ref_8[%val_9] : !p4hir.ref<!arr_4xh>, !b32i
      %val_11 = p4hir.read %b : <!h>
      p4hir.assign %val_11, %elt_ref_10 : <!h>
      %val_12 = p4hir.read %stack : <!hs_4xh>
      %nextIndex_13 = p4hir.struct_extract %val_12["nextIndex"] : !hs_4xh
      %c1_b32i_14 = p4hir.const #int1_b32i
      %sub_15 = p4hir.binop(sub, %nextIndex_13, %c1_b32i_14) : !b32i
      %e = p4hir.variable ["e", init] : <!b32i>
      p4hir.assign %sub_15, %e : <!b32i>
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @c()() {
    p4hir.control_apply {
      %stack = p4hir.variable ["stack"] : <!hs_4xh>
      %hustack = p4hir.variable ["hustack"] : <!hs_2xhu>
      %c3_b32i = p4hir.const #int3_b32i
      %data_field_ref = p4hir.struct_extract_ref %stack["data"] : <!hs_4xh>
      %elt_ref = p4hir.array_element_ref %data_field_ref[%c3_b32i] : !p4hir.ref<!arr_4xh>, !b32i
      %valid = p4hir.const #valid
      %__valid_field_ref = p4hir.struct_extract_ref %elt_ref["__valid"] : <!h>
      p4hir.assign %valid, %__valid_field_ref : <!validity_bit>
      %c3_b32i_0 = p4hir.const #int3_b32i
      %val = p4hir.read %stack : <!hs_4xh>
      %data = p4hir.struct_extract %val["data"] : !hs_4xh
      %array_elt = p4hir.array_get %data[%c3_b32i_0] : !arr_4xh, !b32i
      %b = p4hir.variable ["b", init] : <!h>
      p4hir.assign %array_elt, %b : <!h>
      %c1_b32i = p4hir.const #int1_b32i
      %data_field_ref_1 = p4hir.struct_extract_ref %hustack["data"] : <!hs_2xhu>
      %elt_ref_2 = p4hir.array_element_ref %data_field_ref_1[%c1_b32i] : !p4hir.ref<!arr_2xhu>, !b32i
      %h1_field_ref = p4hir.struct_extract_ref %elt_ref_2["h1"] : <!hu>
      %invalid = p4hir.const #invalid
      %__valid_field_ref_3 = p4hir.struct_extract_ref %h1_field_ref["__valid"] : <!h>
      p4hir.assign %invalid, %__valid_field_ref_3 : <!validity_bit>
      %h2_field_ref = p4hir.struct_extract_ref %elt_ref_2["h2"] : <!hu>
      %invalid_4 = p4hir.const #invalid
      %__valid_field_ref_5 = p4hir.struct_extract_ref %h2_field_ref["__valid"] : <!hh>
      p4hir.assign %invalid_4, %__valid_field_ref_5 : <!validity_bit>
      %h1_field_ref_6 = p4hir.struct_extract_ref %elt_ref_2["h1"] : <!hu>
      %c0_b32i = p4hir.const #int0_b32i
      %val_7 = p4hir.read %hustack : <!hs_2xhu>
      %data_8 = p4hir.struct_extract %val_7["data"] : !hs_2xhu
      %array_elt_9 = p4hir.array_get %data_8[%c0_b32i] : !arr_2xhu, !b32i
      %h1 = p4hir.struct_extract %array_elt_9["h1"] : !hu
      p4hir.assign %h1, %h1_field_ref_6 : <!h>
      %c2_b32i = p4hir.const #int2_b32i
      %data_field_ref_10 = p4hir.struct_extract_ref %stack["data"] : <!hs_4xh>
      %elt_ref_11 = p4hir.array_element_ref %data_field_ref_10[%c2_b32i] : !p4hir.ref<!arr_4xh>, !b32i
      %val_12 = p4hir.read %b : <!h>
      p4hir.assign %val_12, %elt_ref_11 : <!h>
      %c4_b32i = p4hir.const #int4_b32i
      %sz = p4hir.variable ["sz", init] : <!b32i>
      p4hir.assign %c4_b32i, %sz : <!b32i>
    }
  }
}
