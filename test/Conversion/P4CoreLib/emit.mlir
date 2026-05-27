// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s

!b32i = !p4hir.bit<32>
!headers_t = !p4hir.struct<"headers_t">
!packet_out = !p4hir.extern<"packet_out" annotations {corelib}>
!type_T = !p4hir.type_var<"T">
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
module @p4_main {
  // CHECK-NOT: p4hir.extern
  p4hir.extern @packet_out annotations {corelib} {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  // CHECK-LABEL: deparser
  // CHECK-SAME: (%[[arg0:.*]]: !p4corelib.packet_out {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "packet"}, %[[arg1:.*]]: !headers_t {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"})()
  p4hir.control @deparser(%arg0: !packet_out {p4hir.dir = #undir, p4hir.param_name = "packet"}, %arg1: !headers_t {p4hir.dir = #in, p4hir.param_name = "hdr"})() {
    p4hir.control_apply {
      // CHECK: p4corelib.emit %[[arg1]] : !headers_t to %[[arg0]] : !p4corelib.packet_out
      p4hir.call_method @p4_main::@packet_out::@emit<[!headers_t]> (%arg1) of %arg0 : !packet_out : (!headers_t) -> ()
    }
  }
}
