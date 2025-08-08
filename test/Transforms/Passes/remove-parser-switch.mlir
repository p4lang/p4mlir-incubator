// RUN: p4mlir-opt --p4hir-remove-parser-control-flow --canonicalize %s | FileCheck %s

!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>

#int16_b32i = #p4hir.int<16> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int32_b32i = #p4hir.int<32> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int64_b32i = #p4hir.int<64> : !b32i
#int92_b32i = #p4hir.int<92> : !b32i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.parser @p1
  p4hir.parser @p1(%arg0: !b32i, %arg1: !b32i, %arg2: !p4hir.ref<!b8i>)() {
    %c3_b32i = p4hir.const #int3_b32i
    %c2_b32i = p4hir.const #int2_b32i
    %c1_b32i = p4hir.const #int1_b32i
    // CHECK: %temp = p4hir.variable ["temp", init] : <!b32i>
    // CHECK: p4hir.state @start
    p4hir.state @start {
      // CHECK: %add = p4hir.binop(add, %arg0, %arg1) : !b32i
      // CHECK: p4hir.transition_select %add : !b32i {
      %add = p4hir.binop(add, %arg0, %arg1) : !b32i
      %temp = p4hir.variable ["temp", init] : <!b32i>
      p4hir.switch (%add : !b32i) {
        // CHECK: p4hir.select_case
        // CHECK: to @p1::@start_case0
        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
          p4hir.assign %c1_b32i, %temp : <!b32i>
          p4hir.yield
        }

        // CHECK: p4hir.select_case
        // CHECK: to @p1::@start_case1
        p4hir.case(equal, [#int64_b32i]) {
          p4hir.assign %c2_b32i, %temp : <!b32i>
          p4hir.yield
        }

        // CHECK: p4hir.select_case
        // CHECK: p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        // CHECK: to @p1::@start_case2
        p4hir.case(default, [#int92_b32i]) {
          p4hir.assign %c3_b32i, %temp : <!b32i>
          p4hir.yield
        }
        p4hir.yield
      }
      %val = p4hir.read %temp : <!b32i>
      %cast = p4hir.cast(%val : !b32i) : !b8i
      p4hir.assign %cast, %arg2 : <!b8i>
      p4hir.transition to @p1::@accept
    }
    
    // CHECK: p4hir.state @start_case0 {
    // CHECK:   p4hir.assign %c1_b32i, %temp : <!b32i>
    // CHECK:   p4hir.transition to @p1::@start_post
    // CHECK: }
    // CHECK: p4hir.state @start_case1 {
    // CHECK:   p4hir.assign %c2_b32i, %temp : <!b32i>
    // CHECK:   p4hir.transition to @p1::@start_post
    // CHECK: }
    // CHECK: p4hir.state @start_case2 {
    // CHECK:   p4hir.assign %c3_b32i, %temp : <!b32i>
    // CHECK:   p4hir.transition to @p1::@start_post
    // CHECK: }

    // CHECK: p4hir.state @start_post {
    // CHECK:   %val = p4hir.read %temp : <!b32i>
    // CHECK:   %cast = p4hir.cast(%val : !b32i) : !b8i
    // CHECK:   p4hir.assign %cast, %arg2 : <!b8i>
    // CHECK:   p4hir.transition to @p1::@accept
    // CHECK: }

    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p1::@start
  }
}
