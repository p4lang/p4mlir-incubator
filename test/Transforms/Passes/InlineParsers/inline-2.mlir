// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
#everything = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#set_const_of_int2_b8i = #p4hir.set<const : [#int2_b8i]> : !p4hir.set<!b8i>

// CHECK-LABEL: module
module {
  p4hir.parser @callee(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    %set = p4hir.const #set_const_of_int2_b8i
    p4hir.state @start {
      p4hir.transition_select %arg0 : !b8i {
        p4hir.select_case {
          p4hir.yield %set : !p4hir.set<!b8i>
        } to @callee::@reject
        p4hir.select_case {
          %everything = p4hir.const #everything
          p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
        } to @callee::@accept
      }
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee::@start
  }

  // CHECK-LABEL: p4hir.parser @caller
  p4hir.parser @caller(%arg0: !b8i, %arg1: !p4hir.ref<!b8i>)() {
    p4hir.instantiate @callee () as @subparser
    // CHECK-LABEL: p4hir.state @start
    // CHECK:      %[[ADD:.*]] = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
    // CHECK:      p4hir.transition_select %[[ADD]] : !b8i {
    // CHECK-NEXT:   p4hir.select_case {
    // CHECK-NEXT:     p4hir.yield %set : !p4hir.set<!b8i>
    // CHECK-NEXT:   } to @caller::@subparser.reject
    // CHECK-NEXT:   p4hir.select_case {
    // CHECK-NEXT:     p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
    // CHECK-NEXT:   } to @caller::@subparser.accept
    // CHECK-NEXT: }
    p4hir.state @start {
      p4hir.scope {
        %c1_b8i = p4hir.const #int1_b8i
        %add = p4hir.binop(add, %arg0, %c1_b8i) : !b8i
        %ipv4_out_arg = p4hir.variable ["ipv4_out_arg"] : <!b8i>
        p4hir.apply @caller::@subparser(%add, %ipv4_out_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val = p4hir.read %ipv4_out_arg : <!b8i>
        %add_0 = p4hir.binop(add, %val, %c1_b8i) : !b8i
        p4hir.assign %add_0, %arg1 : <!b8i>
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
