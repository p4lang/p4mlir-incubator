// RUN: p4mlir-opt  --p4hir-inline-parsers --p4hir-simplify-parsers --canonicalize %s | FileCheck %s

// Composite parser inlining test, also with constructor arguments.

!b8i = !p4hir.bit<8>
#undir = #p4hir<dir undir>
#calleeWrap_valPassthrough = #p4hir.ctor_param<@calleeWrap, "valPassthrough"> : !b8i
#callee_val = #p4hir.ctor_param<@callee, "val"> : !b8i
#int0_b8i = #p4hir.int<0> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i

// CHECK-LABEL: module
module {
  p4hir.extern @Checksum8 {
    p4hir.func @Checksum8(!b8i {p4hir.dir = #undir, p4hir.param_name = "val"})
  }
  // CHECK-LABEL: p4hir.parser @callee
  p4hir.parser @callee(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "ipv4"})(val: !b8i) {
    %c1_b8i = p4hir.const #int1_b8i
    %val = p4hir.const ["val"] #callee_val
    p4hir.instantiate @Checksum8 (%val : !b8i) as @chksum
    p4hir.state @start {
      %add = p4hir.binop(add, %c1_b8i, %val) : !b8i
      %val_0 = p4hir.read %arg1 : <!b8i>
      %add_1 = p4hir.binop(add, %val_0, %add) : !b8i
      p4hir.assign %add_1, %arg1 : <!b8i>
      p4hir.transition to @callee::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @callee::@start
  }
  // CHECK-LABEL: p4hir.parser @calleeWrap
  p4hir.parser @calleeWrap(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "ipv4"})(valPassthrough: !b8i) {
    // CHECK-DAG: %[[ONE:.*]] = p4hir.const #int1_b8i
    %valPassthrough = p4hir.const ["valPassthrough"] #calleeWrap_valPassthrough
    // CHECK-NOT: p4hir.instantiate @callee
    // CHECK: p4hir.instantiate @Checksum8 (%valPassthrough : !b8i) as @wrap.chksum
    // CHECK-NOT: p4hir.instantiate @callee
    p4hir.instantiate @callee (%valPassthrough : !b8i) as @wrap
    p4hir.state @start {
      p4hir.scope {
        // CHECK-DAG: %[[ADD1:.*]] = p4hir.binop(add, %[[ONE]], %valPassthrough) : !b8i
        // CHECK-DAG: %[[ARG1:.*]] = p4hir.read %ipv4_inout_arg : <!b8i>
        // CHECK-DAG: %{{.*}} = p4hir.binop(add, %[[ARG1]], %[[ADD1]]) : !b8i
        %ipv4_inout_arg = p4hir.variable ["ipv4_inout_arg", init] : <!b8i>
        %val = p4hir.read %arg1 : <!b8i>
        p4hir.assign %val, %ipv4_inout_arg : <!b8i>
        p4hir.apply @calleeWrap::@wrap(%arg0, %ipv4_inout_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_inout_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
      }
      p4hir.transition to @calleeWrap::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @calleeWrap::@start
  }
  // CHECK-LABEL: p4hir.parser @caller
  p4hir.parser @caller(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "ipv4"})() {
    %c1_b8i = p4hir.const #int1_b8i
    %c0_b8i = p4hir.const #int0_b8i
    p4hir.instantiate @calleeWrap (%c0_b8i : !b8i) as @inc1
    p4hir.instantiate @calleeWrap (%c1_b8i : !b8i) as @inc2
    p4hir.state @start {
      p4hir.scope {
        %ipv4_inout_arg = p4hir.variable ["ipv4_inout_arg", init] : <!b8i>
        %val = p4hir.read %arg1 : <!b8i>
        p4hir.assign %val, %ipv4_inout_arg : <!b8i>
        p4hir.apply @caller::@inc1(%arg0, %ipv4_inout_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_inout_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
      }
      p4hir.scope {
        %ipv4_inout_arg = p4hir.variable ["ipv4_inout_arg", init] : <!b8i>
        %val = p4hir.read %arg1 : <!b8i>
        p4hir.assign %val, %ipv4_inout_arg : <!b8i>
        p4hir.apply @caller::@inc2(%arg0, %ipv4_inout_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_inout_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
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
  // CHECK-LABEL: p4hir.parser @caller2
  p4hir.parser @caller2(%arg0: !b8i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "packet"}, %arg1: !p4hir.ref<!b8i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "ipv4"})() {
    // CHECK-DAG: %[[CONST_0:.*]] = p4hir.const #int0_b8i
    // CHECK-DAG: %[[CONST_1:.*]] = p4hir.const #int1_b8i
    // CHECK-DAG: %[[CONST_2:.*]] = p4hir.const #int2_b8i
    // CHECK-DAG: p4hir.instantiate @Checksum8 (%[[CONST_0]] : !b8i) as @subparser.inc1.wrap.chksum
    // CHECK-DAG: p4hir.instantiate @Checksum8 (%[[CONST_1]] : !b8i) as @subparser.inc2.wrap.chksum
    // CHECK-NOT: p4hir.instantiate @caller
    p4hir.instantiate @caller () as @subparser
    p4hir.state @start {
      // CHECK: %{{.*}} = p4hir.binop(add, %{{.*}}, %[[CONST_1]]) : !b8i
      // CHECK: %{{.*}} = p4hir.binop(add, %{{.*}}, %[[CONST_2]]) : !b8i
      // CHECK: %{{.*}} = p4hir.binop(add, %{{.*}}, %[[CONST_1]]) : !b8i
      // CHECK: %{{.*}} = p4hir.binop(add, %{{.*}}, %[[CONST_2]]) : !b8i
      p4hir.scope {
        %ipv4_inout_arg = p4hir.variable ["ipv4_inout_arg", init] : <!b8i>
        %val = p4hir.read %arg1 : <!b8i>
        p4hir.assign %val, %ipv4_inout_arg : <!b8i>
        p4hir.apply @caller2::@subparser(%arg0, %ipv4_inout_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_inout_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
      }
      p4hir.scope {
        %ipv4_inout_arg = p4hir.variable ["ipv4_inout_arg", init] : <!b8i>
        %val = p4hir.read %arg1 : <!b8i>
        p4hir.assign %val, %ipv4_inout_arg : <!b8i>
        p4hir.apply @caller2::@subparser(%arg0, %ipv4_inout_arg) : (!b8i, !p4hir.ref<!b8i>) -> ()
        %val_0 = p4hir.read %ipv4_inout_arg : <!b8i>
        p4hir.assign %val_0, %arg1 : <!b8i>
      }
      p4hir.transition to @caller2::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @caller2::@start
  }
}
