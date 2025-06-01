// RUN: p4mlir-opt %s --verify-roundtrip | FileCheck %s

!headers_t = !p4hir.struct<"headers_t">
module {
  // CHECK-LABEL: deparser
  p4hir.control @deparser(%arg0: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "packet"}, %arg1: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      // CHECK: p4corelib.emit
      p4corelib.emit %arg1 : !headers_t to %arg0 : !p4corelib.packet_out
    }
  }
}

