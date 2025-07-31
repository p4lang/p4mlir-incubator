// RUN: p4mlir-opt --p4hir-simplify-select="flatten-tuples=false replace-bools=false replace-ranges=true concat-args=true" --canonicalize %s | FileCheck %s

!b2i = !p4hir.bit<2>
!b4i = !p4hir.bit<4>
!i8i = !p4hir.int<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#int1_b2i = #p4hir.int<1> : !b2i
#int2_b4i = #p4hir.int<2> : !b4i
#int2_i8i = #p4hir.int<2> : !i8i
#int3_b4i = #p4hir.int<3> : !b4i
#int4_i8i = #p4hir.int<4> : !i8i
#set_const_of_int1_b2i = #p4hir.set<const : [#int1_b2i]> : !p4hir.set<!b2i>
#set_const_of_int2_i8i = #p4hir.set<const : [#int2_i8i]> : !p4hir.set<!i8i>
#set_const_of_int3_b4i = #p4hir.set<const : [#int3_b4i]> : !p4hir.set<!b4i>
#set_const_of_int4_i8i = #p4hir.set<const : [#int4_i8i]> : !p4hir.set<!i8i>
#set_mask_of_int1_b2i_int1_b2i = #p4hir.set<mask : [#int1_b2i, #int1_b2i]> : !p4hir.set<!b2i>
#set_mask_of_int2_b4i_int2_b4i = #p4hir.set<mask : [#int2_b4i, #int2_b4i]> : !p4hir.set<!b4i>

// CHECK-NOT: #p4hir.universal_set

// CHECK-LABEL: module
module {
  p4hir.parser @p1(%arg0: !b2i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "a"}, %arg1: !i8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "b"}, %arg2: !b4i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "c"})() {
    %c3_b4i = p4hir.const #int3_b4i
    %set = p4hir.const #set_mask_of_int2_b4i_int2_b4i
    %set_0 = p4hir.const #set_const_of_int4_i8i
    %set_1 = p4hir.const #set_mask_of_int1_b2i_int1_b2i
    %set_2 = p4hir.const #set_const_of_int3_b4i
    %set_3 = p4hir.const #set_const_of_int2_i8i
    %set_4 = p4hir.const #set_const_of_int1_b2i
    %everything = p4hir.const #everything

    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK: p4hir.concat
      // CHECK: p4hir.transition_select %{{.*}} : !b14i {
      // CHECK-COUNT-4: p4hir.yield %{{.*}} : !p4hir.set<!b14i>
      p4hir.transition_select %arg0, %arg1, %arg2 : !b2i, !i8i, !b4i {
        p4hir.select_case {
          p4hir.yield %set_4, %set_3, %set_2 : !p4hir.set<!b2i>, !p4hir.set<!i8i>, !p4hir.set<!b4i>
        } to @p1::@reject
        p4hir.select_case {
          p4hir.yield %set_1, %set_0, %set : !p4hir.set<!b2i>, !p4hir.set<!i8i>, !p4hir.set<!b4i>
        } to @p1::@reject
        p4hir.select_case {
          %mask = p4hir.mask(%arg2, %c3_b4i) : !p4hir.set<!b4i>
          p4hir.yield %set_1, %everything, %mask : !p4hir.set<!b2i>, !p4hir.set<!p4hir.dontcare>, !p4hir.set<!b4i>
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
