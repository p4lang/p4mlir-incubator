// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Check inlining of extern object.

!b8i = !p4hir.bit<8>
!type_T = !p4hir.type_var<"T">
#in = #p4hir<dir in>
#int1_b8i = #p4hir.int<1> : !b8i
module {
  p4hir.extern @Checksum8 {
    p4hir.func @Checksum8()
    p4hir.func @clear()
    p4hir.func @update<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "data"})
    p4hir.func @remove<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "data"})
    p4hir.func @get() -> !b8i
  }
  p4hir.parser @callee1(%arg0: !b8i {p4hir.dir = #in, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    p4hir.instantiate @Checksum8 () as @chk
    p4hir.state @start {
      %0 = p4hir.call_method @callee1::@chk::@get() : () -> !b8i
      p4hir.assign %0, %arg1 : <!b8i>
      p4hir.transition to @callee1::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee1::@start
  }
  p4hir.parser @callee2(%arg0: !b8i {p4hir.dir = #in, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.instantiate @callee1 () as @subparser1
    p4hir.state @start {
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @callee2::@subparser1(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
      }
      %val = p4hir.read %arg1 : <!b8i>
      %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
      p4hir.assign %add, %arg1 : <!b8i>
      p4hir.transition to @callee2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee2::@start
  }

  // CHECK-LABEL: p4hir.parser @caller
  // CHECK: p4hir.instantiate @Checksum8 () as @subparser2.subparser1.chk
  // CHECK: p4hir.instantiate @Checksum8 () as @subparser3.subparser1.chk
  // CHECK: %0 = p4hir.call_method @caller::@subparser2.subparser1.chk::@get() : () -> !b8i
  // CHECK: p4hir.assign %0, %ipv4_out_arg : <!b8i>
  // CHECK-NOT: p4hir.call_method
  // CHECK-NOT: p4hir.apply
  p4hir.parser @caller(%arg0: !b8i {p4hir.dir = #in, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "ipv4"})() {
    p4hir.instantiate @callee2 () as @subparser2
    p4hir.instantiate @callee2 () as @subparser3
    p4hir.state @start {
      p4hir.scope {
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser2(%arg0, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        p4hir.assign %val, %arg1 : <!b8i>
      }
      p4hir.transition to @caller::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller::@start
  }
}
