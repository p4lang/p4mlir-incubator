// RUN: p4mlir-opt --inline %s | FileCheck %s

!b32i = !p4hir.bit<32>
!top = !p4hir.package<"top">
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#undir = #p4hir<dir undir>
!c = !p4hir.control<"c", (!p4hir.ref<!b32i>)>
!ctr = !p4hir.control<"ctr", (!p4hir.ref<!b32i>)>
module {
  // CHECK-LABEL: p4hir.func @min
  p4hir.func @min(%arg0: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "a"}, %arg1: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "b"}) -> !b32i {
    %false = p4hir.const #false
    %hasReturned = p4hir.variable ["hasReturned", init] : <!p4hir.bool>
    p4hir.assign %false, %hasReturned : <!p4hir.bool>
    %retval = p4hir.variable ["retval"] : <!b32i>
    p4hir.scope {
      %true = p4hir.const #true
      p4hir.assign %true, %hasReturned : <!p4hir.bool>
      %gt = p4hir.cmp(gt, %arg0, %arg1) : !b32i, !p4hir.bool
      %0 = p4hir.ternary(%gt, true {
        p4hir.yield %arg1 : !b32i
      }, false {
        p4hir.yield %arg0 : !b32i
      }) : (!p4hir.bool) -> !b32i
      p4hir.assign %0, %retval : <!b32i>
    }
    %val = p4hir.read %retval : <!b32i>
    p4hir.return %val : !b32i
  }
  // CHECK-LABEL: p4hir.func @fun
  // CHECK-NOT: p4hir.call
  p4hir.func @fun(%arg0: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "a"}, %arg1: !b32i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "b"}) -> !b32i {
    %false = p4hir.const #false
    %hasReturned_0 = p4hir.variable ["hasReturned_0", init] : <!p4hir.bool>
    p4hir.assign %false, %hasReturned_0 : <!p4hir.bool>
    %retval_0 = p4hir.variable ["retval_0"] : <!b32i>
    p4hir.scope {
      %true = p4hir.const #true
      p4hir.assign %true, %hasReturned_0 : <!p4hir.bool>
      %call = p4hir.call @min (%arg0, %arg1) : (!b32i, !b32i) -> !b32i
      %add = p4hir.binop(add, %arg0, %call) : !b32i
      p4hir.assign %add, %retval_0 : <!b32i>
    }
    %val = p4hir.read %retval_0 : <!b32i>
    p4hir.return %val : !b32i
  }
  // CHECK-LABEL: p4hir.control @c
  // CHECK-NOT: p4hir.call
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>, p4hir.param_name = "x"})() {
    p4hir.control_apply {
      %val = p4hir.read %arg0 : <!b32i>
      %val_0 = p4hir.read %arg0 : <!b32i>
      %call = p4hir.call @fun (%val, %val_0) : (!b32i, !b32i) -> !b32i
      p4hir.assign %call, %arg0 : <!b32i>
    }
  }
  p4hir.package @top("_c" : !ctr {p4hir.dir = #undir, p4hir.param_name = "_c"})
  %c = p4hir.instantiate @c() as "c" : () -> !c
  %main = p4hir.instantiate @top(%c) as "main" : (!c) -> !top
}
