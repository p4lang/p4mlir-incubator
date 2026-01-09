// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!anon = !p4hir.enum<a, a_with_control_params>
!b16i = !p4hir.bit<16>
!b8i = !p4hir.bit<8>
!b9i = !p4hir.bit<9>
!validity_bit = !p4hir.validity.bit
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#exact = #p4hir.match_kind<"exact">
#lpm = #p4hir.match_kind<"lpm">
#range = #p4hir.match_kind<"range">
#ternary = #p4hir.match_kind<"ternary">
!Meta_t = !p4hir.struct<"Meta_t", egress_spec: !b9i>
!hdr = !p4hir.header<"hdr", e: !b8i, t: !b16i, l: !b8i, r: !b8i, v: !b8i, __valid: !validity_bit>
!t_exact = !p4hir.struct<"t_exact", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!t_lpm = !p4hir.struct<"t_lpm", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!t_range = !p4hir.struct<"t_range", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
!t_ternary = !p4hir.struct<"t_ternary", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#int-16_b8i = #p4hir.int<240> : !b8i
#int-4081_b16i = #p4hir.int<61455> : !b16i
#int0_b9i = #p4hir.int<0> : !b9i
#int11_b9i = #p4hir.int<11> : !b9i
#int12_b8i = #p4hir.int<12> : !b8i
#int12_b9i = #p4hir.int<12> : !b9i
#int13_b9i = #p4hir.int<13> : !b9i
#int15_b16i = #p4hir.int<15> : !b16i
#int15_b8i = #p4hir.int<15> : !b8i
#int17_b8i = #p4hir.int<17> : !b8i
#int18_b8i = #p4hir.int<18> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int1_b9i = #p4hir.int<1> : !b9i
#int21_b9i = #p4hir.int<21> : !b9i
#int22_b9i = #p4hir.int<22> : !b9i
#int23_b9i = #p4hir.int<23> : !b9i
#int24_b9i = #p4hir.int<24> : !b9i
#int2_b8i = #p4hir.int<2> : !b8i
#int2_b9i = #p4hir.int<2> : !b9i
#int3_b9i = #p4hir.int<3> : !b9i
#int4369_b16i = #p4hir.int<4369> : !b16i
#int4481_b16i = #p4hir.int<4481> : !b16i
#int6_b8i = #p4hir.int<6> : !b8i
#int8_b8i = #p4hir.int<8> : !b8i
!Header_t = !p4hir.struct<"Header_t", h: !hdr>
#set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<mask : [#int17_b8i, #int-16_b8i]> : !p4hir.set<!b8i>
#set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<mask : [#int4369_b16i, #int15_b16i]> : !p4hir.set<!b16i>
#set_mask_of_int4481_b16i_int-4081_b16i = #p4hir.set<mask : [#int4481_b16i, #int-4081_b16i]> : !p4hir.set<!b16i>
#set_range_of_int1_b8i_int8_b8i = #p4hir.set<range : [#int1_b8i, #int8_b8i]> : !p4hir.set<!b8i>
#set_range_of_int6_b8i_int12_b8i = #p4hir.set<range : [#int6_b8i, #int12_b8i]> : !p4hir.set<!b8i>
#set_product_of_set_mask_of_int17_b8i_int-16_b8i = #p4hir.set<product : [#set_mask_of_int17_b8i_int-16_b8i]> : !p4hir.set<tuple<!b8i>>
#set_product_of_set_mask_of_int4369_b16i_int15_b16i = #p4hir.set<product : [#set_mask_of_int4369_b16i_int15_b16i]> : !p4hir.set<tuple<!b16i>>
#set_product_of_set_mask_of_int4481_b16i_int-4081_b16i = #p4hir.set<product : [#set_mask_of_int4481_b16i_int-4081_b16i]> : !p4hir.set<tuple<!b16i>>
#set_product_of_set_range_of_int1_b8i_int8_b8i = #p4hir.set<product : [#set_range_of_int1_b8i_int8_b8i]> : !p4hir.set<tuple<!b8i>>
#set_product_of_set_range_of_int6_b8i_int12_b8i = #p4hir.set<product : [#set_range_of_int6_b8i_int12_b8i]> : !p4hir.set<tuple<!b8i>>
// CHECK: module
module {
  p4hir.control @ingress(%arg0: !p4hir.ref<!Header_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "h"}, %arg1: !p4hir.ref<!Meta_t> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "m"})() {
    p4hir.control_local @__local_ingress_h_0 = %arg0 : !p4hir.ref<!Header_t>
    p4hir.control_local @__local_ingress_m_0 = %arg1 : !p4hir.ref<!Meta_t>
    p4hir.func action @a() {
      %__local_ingress_m_0 = p4hir.symbol_ref @ingress::@__local_ingress_m_0 : !p4hir.ref<!Meta_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %__local_ingress_m_0["egress_spec"] : <!Meta_t>
      %c0_b9i = p4hir.const #int0_b9i
      %cast = p4hir.cast(%c0_b9i : !b9i) : !b9i
      p4hir.assign %cast, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.func action @a_with_control_params(%arg2: !b9i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "x"}) {
      %__local_ingress_m_0 = p4hir.symbol_ref @ingress::@__local_ingress_m_0 : !p4hir.ref<!Meta_t>
      %egress_spec_field_ref = p4hir.struct_field_ref %__local_ingress_m_0["egress_spec"] : <!Meta_t>
      p4hir.assign %arg2, %egress_spec_field_ref : <!b9i>
      p4hir.return
    }
    p4hir.table @t_exact {
      p4hir.table_key(%arg2: !p4hir.ref<!Header_t>) {
        %val = p4hir.read %arg2 : <!Header_t>
        %h = p4hir.struct_extract %val["h"] : !Header_t
        %e = p4hir.struct_extract %h["e"] : !hdr
        p4hir.match_key #exact %e : !b8i annotations {name = "h.h.e"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg2: !b9i {p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg2) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action const {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #p4hir.aggregate<[#int1_b8i]> : tuple<!b8i> {
          %c1_b9i = p4hir.const #int1_b9i
          %cast = p4hir.cast(%c1_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int2_b8i]> : tuple<!b8i> {
          %c2_b9i = p4hir.const #int2_b9i
          %cast = p4hir.cast(%c2_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
      }
    }
    p4hir.table @t_lpm {
      p4hir.table_key(%arg2: !p4hir.ref<!Header_t>) {
        %val = p4hir.read %arg2 : <!Header_t>
        %h = p4hir.struct_extract %val["h"] : !Header_t
        %l = p4hir.struct_extract %h["l"] : !hdr
        p4hir.match_key #lpm %l : !b8i annotations {name = "h.h.l"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg2: !b9i {p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg2) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #set_product_of_set_mask_of_int17_b8i_int-16_b8i {
          %c11_b9i = p4hir.const #int11_b9i
          %cast = p4hir.cast(%c11_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int18_b8i]> : tuple<!b8i> {
          %c12_b9i = p4hir.const #int12_b9i
          %cast = p4hir.cast(%c12_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#everything]> : tuple<!p4hir.set<!p4hir.dontcare>> {
          %c13_b9i = p4hir.const #int13_b9i
          %cast = p4hir.cast(%c13_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
      }
    }
    p4hir.table @t_ternary {
      p4hir.table_key(%arg2: !p4hir.ref<!Header_t>) {
        %val = p4hir.read %arg2 : <!Header_t>
        %h = p4hir.struct_extract %val["h"] : !Header_t
        %t = p4hir.struct_extract %h["t"] : !hdr
        p4hir.match_key #ternary %t : !b16i annotations {name = "h.h.t"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg2: !b9i {p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg2) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #set_product_of_set_mask_of_int4369_b16i_int15_b16i annotations {priority = ["3"]} {
          %c1_b9i = p4hir.const #int1_b9i
          %cast = p4hir.cast(%c1_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int4481_b16i]> : tuple<!b16i> {
          %c2_b9i = p4hir.const #int2_b9i
          %cast = p4hir.cast(%c2_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #set_product_of_set_mask_of_int4481_b16i_int-4081_b16i annotations {priority = ["1"]} {
          %c3_b9i = p4hir.const #int3_b9i
          %cast = p4hir.cast(%c3_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
      }
    }
    p4hir.table @t_range {
      p4hir.table_key(%arg2: !p4hir.ref<!Header_t>) {
        %val = p4hir.read %arg2 : <!Header_t>
        %h = p4hir.struct_extract %val["h"] : !Header_t
        %r = p4hir.struct_extract %h["r"] : !hdr
        p4hir.match_key #range %r : !b8i annotations {name = "h.h.r"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @ingress::@a () : () -> ()
        }
        p4hir.table_action @a_with_control_params(%arg2: !b9i {p4hir.param_name = "x"}) {
          p4hir.call @ingress::@a_with_control_params (%arg2) : (!b9i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @ingress::@a () : () -> ()
      }
      p4hir.table_entries const {
        p4hir.table_entry #set_product_of_set_range_of_int1_b8i_int8_b8i {
          %c21_b9i = p4hir.const #int21_b9i
          %cast = p4hir.cast(%c21_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #set_product_of_set_range_of_int6_b8i_int12_b8i {
          %c22_b9i = p4hir.const #int22_b9i
          %cast = p4hir.cast(%c22_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#int15_b8i]> : tuple<!b8i> {
          %c24_b9i = p4hir.const #int24_b9i
          %cast = p4hir.cast(%c24_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
        p4hir.table_entry #p4hir.aggregate<[#everything]> : tuple<!p4hir.set<!p4hir.dontcare>> {
          %c23_b9i = p4hir.const #int23_b9i
          %cast = p4hir.cast(%c23_b9i : !b9i) : !b9i
          p4hir.call @ingress::@a_with_control_params (%cast) : (!b9i) -> ()
        }
      }
    }
    p4hir.control_apply {
      %t_exact_apply_result = p4hir.table_apply @ingress::@t_exact with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_exact
      %t_lpm_apply_result = p4hir.table_apply @ingress::@t_lpm with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_lpm
      %t_ternary_apply_result = p4hir.table_apply @ingress::@t_ternary with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_ternary
      %t_range_apply_result = p4hir.table_apply @ingress::@t_range with key(%arg0) : (!p4hir.ref<!Header_t>) -> !t_range
    }
  }
}
