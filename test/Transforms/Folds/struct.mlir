// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s -split-input-file | FileCheck %s

!i32i = !p4hir.int<32>
!T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK-LABEL: module
module {
  p4hir.func @blackhole(!i32i)

  %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T

  %t1 = p4hir.struct_extract %t["t1"] : !T

  // CHECK: %[[c10_i32i:.*]] = p4hir.const #int10_i32i
  // CHECK: p4hir.call @blackhole (%[[c10_i32i]])
  p4hir.call @blackhole(%t1) : (!i32i) -> ()

  %var = p4hir.variable ["v"] : <!i32i>
  %v1 = p4hir.read %var : <!i32i>
  %struct = p4hir.struct (%v1, %v1) : !T
  %t11 = p4hir.struct_extract %struct["t1"] : !T

  // CHECK: %[[val:.*]] = p4hir.read %{{.*}} : <!i32i>
  // CHECK: p4hir.call @blackhole (%[[val]])  
  p4hir.call @blackhole(%t11) : (!i32i) -> ()
}

// -----

!b3i = !p4hir.bit<3>
!b8i = !p4hir.bit<8>
!i12i = !p4hir.int<12>
!validity_bit = !p4hir.validity.bit
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool
!A = !p4hir.struct<"A", a: !b8i, b: !i12i, c: !p4hir.bool>
#int0_b8i = #p4hir.int<0> : !b8i
#int0_i12i = #p4hir.int<0> : !i12i
#int1_b3i = #p4hir.int<1> : !b3i
#int1_b8i = #p4hir.int<1> : !b8i
#int2_i12i = #p4hir.int<2> : !i12i
#valid = #p4hir<validity.bit valid> : !validity_bit
!B = !p4hir.header<"B", a: !b3i, b: !A, c: !A, __valid: !validity_bit>

module @p4_main {
  // CHECK-LABEL: p4hir.func @funcA
  p4hir.func @funcA() -> !A {
    // CHECK-DAG: %[[AGG_CST:.*]] = p4hir.const #p4hir.aggregate<[#int1_b8i, #int2_i12i, #true]> : !A
    // CHECK-DAG: p4hir.soft_return %[[AGG_CST]] : !A

    %c1_b8i = p4hir.const #int1_b8i
    %c2_i12i = p4hir.const #int2_i12i
    %true = p4hir.const #true
    %struct_A = p4hir.struct (%c1_b8i, %c2_i12i, %true) : !A
    p4hir.soft_return %struct_A : !A
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @funcB
  p4hir.func @funcB() -> !B {
    // CHECK-DAG: %[[AGG_CST:.*]] = p4hir.const #p4hir.aggregate<[#int1_b3i, #p4hir.aggregate<[#int0_b8i, #int0_i12i, #false]> : !A, #p4hir.aggregate<[#int0_b8i, #int0_i12i, #false]> : !A, #valid]> : !B
    // CHECK-DAG: p4hir.soft_return %[[AGG_CST]] : !B

    %c1_b3i = p4hir.const #int1_b3i
    %c0_b8i = p4hir.const #int0_b8i
    %c0_i12i = p4hir.const #int0_i12i
    %false = p4hir.const #false
    %struct_A = p4hir.struct (%c0_b8i, %c0_i12i, %false) : !A
    %c0_b8i_0 = p4hir.const #int0_b8i
    %c0_i12i_1 = p4hir.const #int0_i12i
    %false_2 = p4hir.const #false
    %struct_A_3 = p4hir.struct (%c0_b8i_0, %c0_i12i_1, %false_2) : !A
    %valid = p4hir.const #valid
    %hdr_B = p4hir.struct (%c1_b3i, %struct_A, %struct_A_3, %valid) : !B
    p4hir.soft_return %hdr_B : !B
    p4hir.return
  }
}
