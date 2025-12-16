// RUN: p4mlir-opt  --p4hir-inline-controls %s | FileCheck %s

// Test inlining argument that has no control local.

!b16i = !p4hir.bit<16>
!validity_bit = !p4hir.validity.bit
!hdr_t = !p4hir.header<"hdr_t", field: !b16i, __valid: !validity_bit>
!headers = !p4hir.struct<"headers", hdr: !hdr_t>

// CHECK-LABEL: module
module {
  p4hir.func action @NoAction() annotations {noWarn = "unused"} {
    p4hir.return
  }
  p4hir.control @MyIngressInner(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"})() annotations {name = "inner_ctrl"} {
    p4hir.control_local @__local_MyIngressInner_hdr_0 = %arg0 : !p4hir.ref<!headers>
    p4hir.func action @set_hdr(%arg1: !b16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) {
      %__local_MyIngressInner_hdr_0 = p4hir.symbol_ref @MyIngressInner::@__local_MyIngressInner_hdr_0 : !p4hir.ref<!headers>
      %hdr_field_ref = p4hir.struct_field_ref %__local_MyIngressInner_hdr_0["hdr"] : <!headers>
      %field_field_ref = p4hir.struct_field_ref %hdr_field_ref["field"] : <!hdr_t>
      p4hir.assign %arg1, %field_field_ref : <!b16i>
      p4hir.return
    }
    p4hir.control_apply {
    }
  }

  // CHECK: p4hir.control @MyIngress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"})() {
  // CHECK:   p4hir.control_local @[[HDR_LOCAL_SYM:.*]] = %arg0 : !p4hir.ref<!headers>
  // CHECK:   p4hir.func action @inner.set_hdr(%[[ACTION_ARG:.*]]: !b16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) {
  // CHECK:     %[[HDR_REF:.*]] = p4hir.symbol_ref @MyIngress::@[[HDR_LOCAL_SYM]] : !p4hir.ref<!headers>
  // CHECK:     %[[FIELD_REF_1:.*]] = p4hir.struct_field_ref %[[HDR_REF]]["hdr"] : <!headers>
  // CHECK:     %[[FIELD_REF_2:.*]] = p4hir.struct_field_ref %[[FIELD_REF_1]]["field"] : <!hdr_t>
  // CHECK:     p4hir.assign %[[ACTION_ARG]], %[[FIELD_REF_2]] : <!b16i>
  // CHECK:     p4hir.return
  // CHECK:   }
  // CHECK:   p4hir.control_apply {
  // CHECK:   }
  // CHECK: }

  p4hir.control @MyIngress(%arg0: !p4hir.ref<!headers> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "hdr"})() {
    p4hir.instantiate @MyIngressInner () as @inner
    p4hir.control_apply {
      p4hir.apply @MyIngress::@inner(%arg0) : (!p4hir.ref<!headers>) -> ()
    }
  }
}
