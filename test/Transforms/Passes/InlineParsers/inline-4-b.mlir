// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Same as test inline-4-a.mlir but parsers have non-conventional order.

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.parser @caller
  p4hir.parser @caller(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    p4hir.instantiate @callee2 () as @subparser2
    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK-DAG: p4hir.assign %c1_b8i, %{{.*}} : <!b8i>
      // CHECK-DAG: p4hir.assign %c2_b8i, %{{.*}} : <!b8i>
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser2(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.parser_accept
    }
    p4hir.transition to @caller::@start
  }
  // CHECK-LABEL: p4hir.parser @callee2
  p4hir.parser @callee2(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c2_b8i = p4hir.const #int2_b8i
    p4hir.instantiate @callee1 () as @subparser1
    p4hir.state @start {
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @callee2::@subparser1(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.assign %c2_b8i, %arg1 : <!b8i>
      p4hir.parser_accept
    }
    p4hir.transition to @callee2::@start
  }
  // CHECK-LABEL: p4hir.parser @callee1
  p4hir.parser @callee1(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      p4hir.assign %c1_b8i, %arg1 : <!b8i>
      p4hir.parser_accept
    }
    p4hir.transition to @callee1::@start
  }

}
