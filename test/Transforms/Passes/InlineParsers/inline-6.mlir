// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Check scope dismantling before inlining.

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i

// CHECK-LABEL: module
// CHECK-NOT: p4hir.scope
module {
  p4hir.parser @callee1(%arg0: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "temp"})() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.state @start {
      %val = p4hir.read %arg0 : <!b8i>
      %add = p4hir.binop(add, %val, %c1_b8i) : !b8i
      p4hir.assign %add, %arg0 : <!b8i>
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
  p4hir.parser @caller1()() {
    %c1_b8i = p4hir.const #int1_b8i
    p4hir.instantiate @callee1 () as @subparser
    p4hir.state @start {
      p4hir.scope {
        p4hir.scope {
          %temp = p4hir.variable ["temp", init] : <!b8i>
          p4hir.assign %c1_b8i, %temp : <!b8i>
          p4hir.scope {
            %temp_inout_arg = p4hir.variable ["temp_inout_arg", init] : <!b8i>
            %val = p4hir.read %temp : <!b8i>
            p4hir.assign %val, %temp_inout_arg : <!b8i>
            p4hir.apply @caller1::@subparser(%temp_inout_arg) : (!p4hir.ref<!b8i>) -> ()
            %val_0 = p4hir.read %temp_inout_arg : <!b8i>
            p4hir.assign %val_0, %temp : <!b8i>
          }
        }
      }
      p4hir.transition to @caller1::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller1::@start
  }
}
