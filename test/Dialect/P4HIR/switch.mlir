// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!b32i = !p4hir.bit<32>
!top = !p4hir.package<"top">
#undir = #p4hir<dir undir>
#int16_b32i = #p4hir.int<16> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int32_b32i = #p4hir.int<32> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int64_b32i = #p4hir.int<64> : !b32i
#int92_b32i = #p4hir.int<92> : !b32i
!c = !p4hir.control<"c", (!p4hir.ref<!b32i>)>
!ct = !p4hir.control<"ct", (!p4hir.ref<!b32i>)>
// CHECK: module
module {
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "b"})() {
    p4hir.control_apply {
      %val = p4hir.read %arg0 : <!b32i>
      p4hir.switch (%val : !b32i) {
        p4hir.case(anyof, [#int16_b32i, #int32_b32i]) {
          %c1_b32i = p4hir.const #int1_b32i
          p4hir.assign %c1_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(equal, [#int64_b32i]) {
          %c2_b32i = p4hir.const #int2_b32i
          p4hir.assign %c2_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.case(default, [#int92_b32i]) {
          %c3_b32i = p4hir.const #int3_b32i
          p4hir.assign %c3_b32i, %arg0 : <!b32i>
          p4hir.yield
        }
        p4hir.yield
      }
    }
  }
  p4hir.package @top("_c" : !ct {p4hir.dir = #undir, p4hir.param_name = "_c"})
  %c = p4hir.instantiate @c() as "c" : () -> !c
  %main = p4hir.instantiate @top(%c) as "main" : (!c) -> !top
}
