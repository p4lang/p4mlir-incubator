// RUN: p4mlir-opt --canonicalize %s | FileCheck %s
!b32i = !p4hir.bit<32>
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i

module {
  p4hir.func @f1(%arg0: !p4hir.ref<!b32i>) {
    %val = p4hir.read %arg0 : <!b32i>
    // CHECK-NOT: p4hir.switch
  p4hir.switch (%val : !b32i) {
    p4hir.case(equal, [#int1_b32i]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int2_b32i]) {
      p4hir.yield
    }
    p4hir.case(equal, [#int3_b32i]) {
      p4hir.yield
    }
    p4hir.case(default, []) {
      p4hir.yield
    }
    p4hir.yield
  }
  p4hir.return
}}
