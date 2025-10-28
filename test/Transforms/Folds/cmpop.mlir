// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
!b32i = !p4hir.bit<32>
!int = !p4hir.infint

#int5_i32i = #p4hir.int<5> : !i32i
#int10_i32i = #p4hir.int<10> : !i32i
#int15_i32i = #p4hir.int<15> : !i32i
#int0_b32i = #p4hir.int<0> : !b32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_bool(!p4hir.bool)
  p4hir.func @blackhole_b32i(!b32i)
  p4hir.func @blackhole_int(!int)

  // CHECK: p4hir.func @test(%[[ARG:.*]]: !i32i)
  p4hir.func @test(%arg_i32i : !i32i) {
    // CHECK: %[[c5_i32i:.*]] = p4hir.const #int5_i32i
    %c5_i32i = p4hir.const #int5_i32i
    %c10_i32i = p4hir.const #int10_i32i
    %c15_i32i = p4hir.const #int15_i32i
    %c0_b32i = p4hir.const #int0_b32i

    %c255_int = p4hir.const #p4hir.int<255> : !int
    %c45506_int = p4hir.const #p4hir.int<45506> : !int

    // EQ

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res1_eq = p4hir.cmp(eq, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_eq) : (!p4hir.bool) -> ()

    // CHECK: %[[EQ:.*]] = p4hir.cmp(eq, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[EQ]])
    %res2_eq = p4hir.cmp(eq, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_eq) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res3_eq = p4hir.cmp(eq, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_eq) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res4_eq = p4hir.cmp(eq, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_eq) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res5_eq = p4hir.cmp(eq, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_eq) : (!p4hir.bool) -> ()

    // NE

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res1_ne = p4hir.cmp(ne, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_ne) : (!p4hir.bool) -> ()

    // CHECK: %[[NE:.*]] = p4hir.cmp(ne, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[NE]])
    %res2_ne = p4hir.cmp(ne, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_ne) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res3_ne = p4hir.cmp(ne, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_ne) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res4_ne = p4hir.cmp(ne, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_ne) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res5_ne = p4hir.cmp(ne, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_ne) : (!p4hir.bool) -> ()

    // LT

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res1_lt = p4hir.cmp(lt, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_lt) : (!p4hir.bool) -> ()

    // CHECK: %[[GT:.*]] = p4hir.cmp(gt, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[GT]])
    %res2_lt = p4hir.cmp(lt, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_lt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res3_lt = p4hir.cmp(lt, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_lt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res4_lt = p4hir.cmp(lt, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_lt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res5_lt = p4hir.cmp(lt, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_lt) : (!p4hir.bool) -> ()

    // LE

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res1_le = p4hir.cmp(le, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_le) : (!p4hir.bool) -> ()

    // CHECK: %[[GE:.*]] = p4hir.cmp(ge, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[GE]])
    %res2_le = p4hir.cmp(le, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_le) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res3_le = p4hir.cmp(le, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_le) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res4_le = p4hir.cmp(le, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_le) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res5_le = p4hir.cmp(le, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_le) : (!p4hir.bool) -> ()

    // GT

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res1_gt = p4hir.cmp(gt, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_gt) : (!p4hir.bool) -> ()

    // CHECK: %[[LT:.*]] = p4hir.cmp(lt, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[LT]])
    %res2_gt = p4hir.cmp(gt, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_gt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res3_gt = p4hir.cmp(gt, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_gt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%false)
    %res4_gt = p4hir.cmp(gt, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_gt) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res5_gt = p4hir.cmp(gt, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_gt) : (!p4hir.bool) -> ()

    // GE

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res1_ge = p4hir.cmp(ge, %arg_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res1_ge) : (!p4hir.bool) -> ()

    // CHECK: %[[LE:.*]] = p4hir.cmp(le, %[[ARG]] : !i32i, %[[c5_i32i]] : !i32i)
    // CHECK: p4hir.call @blackhole_bool (%[[LE]])
    %res2_ge = p4hir.cmp(ge, %c5_i32i : !i32i, %arg_i32i : !i32i)
    p4hir.call @blackhole_bool(%res2_ge) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res3_ge = p4hir.cmp(ge, %c15_i32i : !i32i, %c10_i32i : !i32i)
    p4hir.call @blackhole_bool(%res3_ge) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res4_ge = p4hir.cmp(ge, %c0_b32i : !b32i, %c0_b32i : !b32i)
    p4hir.call @blackhole_bool(%res4_ge) : (!p4hir.bool) -> ()

    // CHECK: p4hir.call @blackhole_bool (%true)
    %res5_ge = p4hir.cmp(ge, %c45506_int : !int, %c255_int : !int)
    p4hir.call @blackhole_bool(%res5_ge) : (!p4hir.bool) -> ()

    p4hir.return
  }
}
