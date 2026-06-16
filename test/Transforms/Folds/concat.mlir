// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s -split-input-file | FileCheck %s

!b3i = !p4hir.bit<3>
!b5i = !p4hir.bit<5>
!b6i = !p4hir.bit<6>
!b7i = !p4hir.bit<7>
!b8i = !p4hir.bit<8>
!i4i = !p4hir.int<4>
!i7i = !p4hir.int<7>
!i8i = !p4hir.int<8>

#int6_b3i = #p4hir.int<6> : !b3i
#int5_b5i = #p4hir.int<5> : !b5i
#int-3_i4i = #p4hir.int<-3> : !i4i

// CHECK-DAG: #[[ATTR_197:.+]] = #p4hir.int<197> : !b8i
// CHECK-DAG: #[[ATTR_M18:.+]] = #p4hir.int<-18> : !i7i
// CHECK-DAG: #[[ATTR_54:.+]] = #p4hir.int<54> : !b6i
// CHECK-DAG: #[[ATTR_109:.+]] = #p4hir.int<109> : !b7i
// CHECK-DAG: #[[ATTR_M35:.+]] = #p4hir.int<-35> : !i8i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole_b6i(!b6i)
  p4hir.func @blackhole_b7i(!b7i)
  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_i7i(!i7i)
  p4hir.func @blackhole_i8i(!i8i)

  // CHECK-LABEL: p4hir.func @test_concat_const
  p4hir.func @test_concat_const() {
    // CHECK-DAG: %[[C197:.*]] = p4hir.const #[[ATTR_197]]
    // CHECK-DAG: %[[CM18:.*]] = p4hir.const #[[ATTR_M18]]
    // CHECK-DAG: %[[C54:.*]] = p4hir.const #[[ATTR_54]]
    // CHECK-DAG: %[[C109:.*]] = p4hir.const #[[ATTR_109]]
    // CHECK-DAG: %[[CM35:.*]] = p4hir.const #[[ATTR_M35]]

    %c6_b3i = p4hir.const #int6_b3i
    %c5_b5i = p4hir.const #int5_b5i
    %c_3_i4i = p4hir.const #int-3_i4i

    // CHECK: p4hir.call @blackhole_b8i (%[[C197]])
    %r1 = p4hir.concat(%c6_b3i : !b3i, %c5_b5i : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r1) : (!b8i) -> ()

    // Signed lhs: concat(0b1101, 0b110) = 0b1101110 = -18 as int<7>
    // CHECK: p4hir.call @blackhole_i7i (%[[CM18]])
    %r2 = p4hir.concat(%c_3_i4i : !i4i, %c6_b3i : !b3i) : !i7i
    p4hir.call @blackhole_i7i(%r2) : (!i7i) -> ()

    // Same-width operands: concat(0b110, 0b110) = 0b110110 = 54
    // CHECK: p4hir.call @blackhole_b6i (%[[C54]])
    %r3 = p4hir.concat(%c6_b3i : !b3i, %c6_b3i : !b3i) : !b6i
    p4hir.call @blackhole_b6i(%r3) : (!b6i) -> ()

    // Unsigned lhs, signed rhs: concat(0b110, 0b1101) = 0b1101101 = 109 as bit<7>
    // CHECK: p4hir.call @blackhole_b7i (%[[C109]])
    %r4 = p4hir.concat(%c6_b3i : !b3i, %c_3_i4i : !i4i) : !b7i
    p4hir.call @blackhole_b7i(%r4) : (!b7i) -> ()

    // Both signed: concat(0b1101, 0b1101) = 0b11011101 = -35 as int<8>
    // CHECK: p4hir.call @blackhole_i8i (%[[CM35]])
    %r5 = p4hir.concat(%c_3_i4i : !i4i, %c_3_i4i : !i4i) : !i8i
    p4hir.call @blackhole_i8i(%r5) : (!i8i) -> ()

    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_concat_no_fold
  p4hir.func @test_concat_no_fold(%arg0: !b3i, %arg1: !b5i) {
    // CHECK: p4hir.concat
    %r = p4hir.concat(%arg0 : !b3i, %arg1 : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r) : (!b8i) -> ()
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @test_concat_partial_const
  p4hir.func @test_concat_partial_const(%arg0: !b3i) {
    %c5_b5i = p4hir.const #int5_b5i
    // CHECK: p4hir.concat
    %r = p4hir.concat(%arg0 : !b3i, %c5_b5i : !b5i) : !b8i
    p4hir.call @blackhole_b8i(%r) : (!b8i) -> ()
    p4hir.return
  }
}

// -----

!b10i = !p4hir.bit<10>
!b16i = !p4hir.bit<16>
!b3i = !p4hir.bit<3>
!b5i = !p4hir.bit<5>
!b6i = !p4hir.bit<6>
!i6i = !p4hir.int<6>
!b7i = !p4hir.bit<7>
!b9i = !p4hir.bit<9>
!i10i = !p4hir.int<10>
!b11i = !p4hir.bit<11>
!i11i = !p4hir.int<11>
!i16i = !p4hir.int<16>
!i5i = !p4hir.int<5>
module {
  // CHECK-LABEL: p4hir.func @funcA
  p4hir.func @funcA(%arg0: !b16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !b10i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !b16i -> !b10i
    // CHECK-DAG: p4hir.soft_return %[[NEW_SLICE]] : !b10i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !b16i -> !b5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !b16i -> !b5i
    %0 = p4hir.concat(%s13_9 : !b5i, %s8_4 : !b5i) : !b10i
    p4hir.soft_return %0 : !b10i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcB
  p4hir.func @funcB(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !b10i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !i16i -> !b10i
    // CHECK-DAG: p4hir.soft_return %[[NEW_SLICE]] : !b10i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %0 = p4hir.concat(%s13_9 : !b5i, %s8_4 : !b5i) : !b10i
    p4hir.soft_return %0 : !b10i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcC
  p4hir.func @funcC(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !i10i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !i16i -> !b10i
    // CHECK-DAG: %[[CAST:.*]] = p4hir.cast(%[[NEW_SLICE]] : !b10i) : !i10i
    // CHECK-DAG: p4hir.soft_return %[[CAST]] : !i10i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %cast = p4hir.cast(%s13_9 : !b5i) : !i5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %0 = p4hir.concat(%cast : !i5i, %s8_4 : !b5i) : !i10i
    p4hir.soft_return %0 : !i10i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcD
  p4hir.func @funcD(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !b10i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !i16i -> !b10i
    // CHECK-DAG: p4hir.soft_return %[[NEW_SLICE]] : !b10i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %cast = p4hir.cast(%s8_4 : !b5i) : !i5i
    %0 = p4hir.concat(%s13_9 : !b5i, %cast : !i5i) : !b10i
    p4hir.soft_return %0 : !b10i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcE
  p4hir.func @funcE(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !i10i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !i16i -> !b10i
    // CHECK-DAG: %[[CAST:.*]] = p4hir.cast(%[[NEW_SLICE]] : !b10i) : !i10i
    // CHECK-DAG: p4hir.soft_return %[[CAST]] : !i10i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %cast = p4hir.cast(%s13_9 : !b5i) : !i5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %cast_0 = p4hir.cast(%s8_4 : !b5i) : !i5i
    %0 = p4hir.concat(%cast : !i5i, %cast_0 : !i5i) : !i10i
    p4hir.soft_return %0 : !i10i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcF
  p4hir.func @funcF(%arg0: !b16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !b16i {
    // CHECK-DAG: p4hir.soft_return %arg0 : !b16i

    %s15_13 = p4hir.slice %arg0[15 : 13] : !b16i -> !b3i
    %s12_7 = p4hir.slice %arg0[12 : 7] : !b16i -> !b6i
    %0 = p4hir.concat(%s15_13 : !b3i, %s12_7 : !b6i) : !b9i
    %s6_0 = p4hir.slice %arg0[6 : 0] : !b16i -> !b7i
    %1 = p4hir.concat(%0 : !b9i, %s6_0 : !b7i) : !b16i
    p4hir.soft_return %1 : !b16i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcG
  p4hir.func @funcG(%arg0: !b16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !b11i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !b16i -> !b10i
    // CHECK-DAG: %[[CAST:.*]] = p4hir.cast(%[[NEW_SLICE]] : !b10i) : !b11i
    // CHECK-DAG: p4hir.soft_return %[[CAST]] : !b11i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !b16i -> !b5i
    %cast = p4hir.cast(%s13_9 : !b5i) : !b6i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !b16i -> !b5i
    %0 = p4hir.concat(%cast : !b6i, %s8_4 : !b5i) : !b11i
    p4hir.soft_return %0 : !b11i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcH
  p4hir.func @funcH(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !i11i {
    // Cannot optimize with RHS cast that changes width.
    // CHECK-COUNT-2: p4hir.slice
    // CHECK: p4hir.concat

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %cast = p4hir.cast(%s13_9 : !b5i) : !i5i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %cast_0 = p4hir.cast(%s8_4 : !b5i) : !b6i
    %cast_1 = p4hir.cast(%cast_0 : !b6i) : !i6i
    %0 = p4hir.concat(%cast : !i5i, %cast_1 : !i6i) : !i11i
    p4hir.soft_return %0 : !i11i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcI
  p4hir.func @funcI(%arg0: !i16i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "val"}) -> !i11i {
    // CHECK-DAG: %[[NEW_SLICE:.*]] = p4hir.slice %arg0[13 : 4] : !i16i -> !b10i
    // CHECK-DAG: %[[CAST_1:.*]] = p4hir.cast(%[[NEW_SLICE]] : !b10i) : !i10i
    // CHECK-DAG: %[[CAST_2:.*]] = p4hir.cast(%[[CAST_1]] : !i10i) : !i11i
    // CHECK-DAG: p4hir.soft_return %[[CAST_2]] : !i11i

    %s13_9 = p4hir.slice %arg0[13 : 9] : !i16i -> !b5i
    %cast = p4hir.cast(%s13_9 : !b5i) : !b6i
    %cast_0 = p4hir.cast(%cast : !b6i) : !i6i
    %s8_4 = p4hir.slice %arg0[8 : 4] : !i16i -> !b5i
    %cast_1 = p4hir.cast(%s8_4 : !b5i) : !i5i
    %0 = p4hir.concat(%cast_0 : !i6i, %cast_1 : !i5i) : !i11i
    p4hir.soft_return %0 : !i11i
    p4hir.return
  }
}
