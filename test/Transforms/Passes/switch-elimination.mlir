// RUN: p4mlir-opt --p4hir-switch-elimination %s | FileCheck %s

!b32i = !p4hir.bit<32>
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int16_b32i = #p4hir.int<16> : !b32i
#int32_b32i = #p4hir.int<32> : !b32i

// CHECK-LABEL: p4hir.control @c
module {
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "b"})() {
    // CHECK: p4hir.func action @_switch_0_case_0
    // CHECK: p4hir.func action @_switch_0_case_1
    // CHECK: p4hir.func action @_switch_0_default
    // CHECK: p4hir.table @_switch_0_table
    p4hir.control_apply {
      %val = p4hir.read %arg0 : <!b32i>
      // CHECK: %[[KEY:.*]] = p4hir.variable ["_switch_0_key"
      // CHECK: p4hir.assign %{{.*}}, %[[KEY]]
      // CHECK: %[[KEY_VAL:.*]] = p4hir.read %[[KEY]]
      // CHECK: p4hir.table_apply @c::@_switch_0_table with key(%[[KEY_VAL]])
      // CHECK: p4hir.struct_extract
      // CHECK: p4hir.switch
      p4hir.switch (%val : !b32i) {
        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
          %c1_b32i = p4hir.const #int1_b32i
          p4hir.assign %c1_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(equal, [#int2_b32i]) {
          %c2_b32i = p4hir.const #int2_b32i
          p4hir.assign %c2_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(default, []) {
          %c3_b32i = p4hir.const #int3_b32i
          p4hir.assign %c3_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.yield
      }
    }
  }
}
