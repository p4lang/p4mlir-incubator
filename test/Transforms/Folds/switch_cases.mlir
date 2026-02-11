// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b32 = !p4hir.bit<32>
!ref_b32 = !p4hir.ref<!b32>
#int1_b32 = #p4hir.int<1> : !b32
#int2_b32 = #p4hir.int<2> : !b32
#int3_b32 = #p4hir.int<3> : !b32
#int4_b32 = #p4hir.int<4> : !b32
#int5_b32 = #p4hir.int<5> : !b32

// CHECK-LABEL: @test_merge_empty_cases
p4hir.func @test_merge_empty_cases(%val: !b32, %ref: !ref_b32) {
  // Test that empty cases with empty default are removed
  // CHECK: p4hir.switch
  // CHECK: p4hir.case(equal, [#int4_b32i])
  // CHECK: p4hir.assign
  // CHECK: p4hir.case(default
  p4hir.switch (%val : !b32) {
    p4hir.case(equal, [#int1_b32]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int2_b32]) {
      p4hir.yield
    }
    p4hir.case(anyof, [#int3_b32]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int4_b32]) {
      %c = p4hir.const #int5_b32
      p4hir.assign %c, %ref : <!b32>
      p4hir.yield
    }
    p4hir.case(default, []) {
      p4hir.yield
    }
    p4hir.yield
  }
  p4hir.return
}

// CHECK-LABEL: @test_non_empty_default
p4hir.func @test_non_empty_default(%val: !b32, %ref: !ref_b32) {
  // Test that empty cases ARE merged even when default is non-empty
  // CHECK: p4hir.switch
  // CHECK: p4hir.case(anyof, [#int1_b32i, #int2_b32i])
  // CHECK-NEXT: p4hir.yield
  // CHECK-NEXT: }
  // CHECK: p4hir.case(default
  p4hir.switch (%val : !b32) {
    p4hir.case(equal, [#int1_b32]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int2_b32]) {
      p4hir.yield
    }
    p4hir.case(default, []) {
      %c = p4hir.const #int5_b32
      p4hir.assign %c, %ref : <!b32>
      p4hir.yield
    }
    p4hir.yield
  }
  p4hir.return
}

// CHECK-LABEL: @test_mixed_empty_non_empty
p4hir.func @test_mixed_empty_non_empty(%val: !b32, %ref: !ref_b32) {
  // Test with mix of empty and non-empty cases with empty default
  // Empty cases are removed when default is empty
  // CHECK: p4hir.const
  // CHECK: p4hir.switch
  // CHECK: p4hir.case(equal, [#int2_b32i])
  // CHECK: p4hir.assign
  // CHECK: p4hir.case(default
  p4hir.switch (%val : !b32) {
    p4hir.case(equal, [#int1_b32]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int2_b32]) {
      %c = p4hir.const #int4_b32
      p4hir.assign %c, %ref : <!b32>
      p4hir.yield
    }
    p4hir.case(equal, [#int3_b32]) {
      p4hir.yield
    }
    p4hir.case(default, []) {
      p4hir.yield
    }
    p4hir.yield
  }
  p4hir.return
}

// CHECK-LABEL: @test_no_default
p4hir.func @test_no_default(%val: !b32, %ref: !ref_b32) {
  // Test with no default case - empty cases are removed
  // CHECK: p4hir.const
  // CHECK: p4hir.switch
  // CHECK-NOT: p4hir.case(anyof
  // CHECK: p4hir.case(equal, [#int2_b32i])
  // CHECK: p4hir.assign
  // CHECK-NOT: p4hir.case(equal, [#int1_b32i])
  // CHECK-NOT: p4hir.case(equal, [#int3_b32i])
  p4hir.switch (%val : !b32) {
    p4hir.case(equal, [#int1_b32]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int2_b32]) {
      %c = p4hir.const #int4_b32
      p4hir.assign %c, %ref : <!b32>
      p4hir.yield
    }
    p4hir.case(equal, [#int3_b32]) {
      p4hir.yield
    }
    p4hir.yield
  }
  p4hir.return
}
