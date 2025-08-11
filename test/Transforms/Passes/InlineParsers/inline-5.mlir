// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Check "diamond-shaped" inlining:
//
// p4hir.parser @callee1 {
//   ...
//   mark1;
// }
// p4hir.parser @callee2 {
//   callee1() subparser1;
//   ...
//     subparser1.apply(...)
//     mark2;
// }
// p4hir.parser @callee3 {
//   callee1() subparser1;
//   ...
//     subparser1.apply(...)
//     mark3;
// }
// p4hir.parser @caller {
//   callee2() subparser2;
//   callee3() subparser3;
//   ...
//     subparser2.apply(...)
//     subparser3.apply(...)
// }
//
// After inlining, caller must be like:
// p4hir.parser @caller {
//   ...
//   mark1
//   mark2
//   mark1
//   mark3
// }

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int3_b8i = #p4hir.int<3> : !b8i

// CHECK-LABEL: module
module {
  p4hir.parser @callee1(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      p4hir.assign %c1_b8i, %arg1 : <!b8i>
      p4hir.parser_accept
    }
    p4hir.transition to @callee1::@start
  }
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
  p4hir.parser @callee3(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c3_b8i = p4hir.const #int3_b8i
    p4hir.instantiate @callee1 () as @subparser1
    p4hir.state @start {
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @callee3::@subparser1(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.assign %c3_b8i, %arg1 : <!b8i>
      p4hir.parser_accept
    }
    p4hir.transition to @callee3::@start
  }
  // CHECK-LABEL: p4hir.parser @caller
  p4hir.parser @caller(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    p4hir.instantiate @callee2 () as @subparser2
    p4hir.instantiate @callee3 () as @subparser3
    // CHECK-LABEL: p4hir.state @start
    p4hir.state @start {
      // CHECK: p4hir.assign %c1_b8i, %{{.*}} : <!b8i>
      // CHECK: p4hir.assign %c2_b8i, %{{.*}} : <!b8i>
      // CHECK: p4hir.assign %c1_b8i, %{{.*}} : <!b8i>
      // CHECK: p4hir.assign %c3_b8i, %{{.*}} : <!b8i>
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser2(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser3(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.parser_accept
    }
    p4hir.transition to @caller::@start
  }
}
