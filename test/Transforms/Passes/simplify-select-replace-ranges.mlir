// RUN: p4mlir-opt --p4hir-simplify-select="flatten-tuples=false replace-bools=false replace-ranges=true concat-args=false" --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#int0_b8i = #p4hir.int<0> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i
#int1_i8i = #p4hir.int<1> : !i8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i
#int3_i8i = #p4hir.int<3> : !i8i
#int8_b8i = #p4hir.int<8> : !b8i
#int8_i8i = #p4hir.int<8> : !i8i
#set_const_of_int8_b8i = #p4hir.set<const : [#int8_b8i]> : !p4hir.set<!b8i>
#set_mask_of_int1_i8i_int1_i8i = #p4hir.set<mask : [#int1_i8i, #int1_i8i]> : !p4hir.set<!i8i>
#set_range_of_int0_b8i_int3_b8i = #p4hir.set<range : [#int0_b8i, #int3_b8i]> : !p4hir.set<!b8i>
#set_range_of_int1_b8i_int2_b8i = #p4hir.set<range : [#int1_b8i, #int2_b8i]> : !p4hir.set<!b8i>
#set_range_of_int1_i8i_int3_i8i = #p4hir.set<range : [#int1_i8i, #int3_i8i]> : !p4hir.set<!i8i>
#set_range_of_int3_i8i_int8_i8i = #p4hir.set<range : [#int3_i8i, #int8_i8i]> : !p4hir.set<!i8i>

// CHECK-LABEL: module
module {
  p4hir.parser @p1(%arg0: !i8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "a"}, %arg1: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "b"})() {
    // CHECK: %[[cst0:set.*]] = p4hir.const #set_mask_of_int0_b8i_int-4_b8i
    // CHECK: %[[cst1:set.*]] = p4hir.const #set_mask_of_int1_b8i_int1_b8i
    // CHECK: %[[cst2:set.*]] = p4hir.const #set_mask_of_int8_b8i_int-1_b8i
    // CHECK: %[[cst3:set.*]] = p4hir.const #set_mask_of_int4_b8i_int-4_b8i
    // CHECK: %[[cst4:set.*]] = p4hir.const #set_mask_of_int2_b8i_int-1_b8i
    // CHECK: %[[cst5:set.*]] = p4hir.const #set_mask_of_int3_b8i_int-1_b8i
    // CHECK: %[[cst6:set.*]] = p4hir.const #set_mask_of_int2_b8i_int-2_b8i
    // CHECK: %[[cst7:set.*]] = p4hir.const #set_mask_of_int1_b8i_int-1_b8i
    // CHECK: %[[cst8:set.*]] = p4hir.const #set_const_of_int8_b8i
    %set = p4hir.const #set_range_of_int0_b8i_int3_b8i
    %set_0 = p4hir.const #set_mask_of_int1_i8i_int1_i8i
    %set_1 = p4hir.const #set_range_of_int1_b8i_int2_b8i
    %set_2 = p4hir.const #set_range_of_int3_i8i_int8_i8i
    %set_3 = p4hir.const #set_const_of_int8_b8i
    %set_4 = p4hir.const #set_range_of_int1_i8i_int3_i8i
    %everything = p4hir.const #everything

    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK: %[[cast:.*]] = p4hir.cast(%arg0 : !i8i) : !b8i
      p4hir.transition_select %arg0, %arg1 : !i8i, !b8i {
        p4hir.select_case {
          p4hir.yield %set_4, %set_3 : !p4hir.set<!i8i>, !p4hir.set<!b8i>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %set_2, %set_1 : !p4hir.set<!i8i>, !p4hir.set<!b8i>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %set_0, %set : !p4hir.set<!i8i>, !p4hir.set<!b8i>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @p1::@accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p1::@start
  }
}
