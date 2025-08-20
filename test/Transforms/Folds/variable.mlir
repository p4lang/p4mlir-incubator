// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b1i = !p4hir.bit<1>
!b8i = !p4hir.bit<8>
!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
#int100_b8i = #p4hir.int<100> : !b8i
#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  // CHECK-LABEL: p4hir.func @test_write_once
  p4hir.func @test_write_once() {
    %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T

    %var = p4hir.variable ["v"] : <!T>
    p4hir.assign %t, %var : <!T>

    %struct = p4hir.read %var : <!T>
    %t11 = p4hir.struct_extract %struct["t1"] : !T

    // This all just simplifies down to constant
    // CHECK: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
    // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
    p4hir.call @blackhole(%t11) : (!i32i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_write_only
  p4hir.func @test_write_only(%arg0: !p4hir.ref<!b8i>) {
    // CHECK-NOT: p4hir.variable
    // CHECK: p4hir.return
    %return_guard = p4hir.variable ["return_guard"] : <!p4hir.bool>
    %true = p4hir.const #true
    p4hir.assign %true, %return_guard : <!p4hir.bool>
    %c2_b8i = p4hir.const #int2_b8i
    %c1_b8i = p4hir.const #int1_b8i
    %c100_b8i = p4hir.const #int100_b8i
    %val = p4hir.read %arg0 : <!b8i>
    %gt = p4hir.cmp(gt, %val : !b8i, %c100_b8i : !b8i)
    p4hir.if %gt {
      %false = p4hir.const #false
      p4hir.assign %false, %return_guard : <!p4hir.bool>
    } else {
      %false = p4hir.const #false
      p4hir.assign %false, %return_guard : <!p4hir.bool>
    }

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_read_before_write
  p4hir.func @test_read_before_write() {
    // Check that we don't crash.
    %a = p4hir.variable ["a"] : <!b1i>
    %d = p4hir.variable ["d"] : <!p4hir.bool>
    %val = p4hir.read %d : <!p4hir.bool>
    %cast = p4hir.cast(%val : !p4hir.bool) : !b1i
    p4hir.assign %cast, %a : <!b1i>
    %val_0 = p4hir.read %a : <!b1i>
    %cast_1 = p4hir.cast(%val_0 : !b1i) : !p4hir.bool
    p4hir.assign %cast_1, %d : <!p4hir.bool>

    p4hir.return
  }
}
