// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i

// CHECK-LABEL: module
module {
  p4hir.parser @callee(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      p4hir.assign %c1_b8i, %arg1 : <!b8i>
      p4hir.parser_accept
    }
    p4hir.transition to @callee::@start
  }

  // CHECK-LABEL: p4hir.parser @caller
  p4hir.parser @caller(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    p4hir.instantiate @callee () as @subparser
    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // Test proper inlining of code in an accept state.
      // CHECK: p4hir.assign %c1_b8i, %arg1 : <!b8i>
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.parser_accept
    }
    p4hir.transition to @caller::@start
  }
}
