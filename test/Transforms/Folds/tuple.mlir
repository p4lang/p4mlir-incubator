// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b8i = !p4hir.bit<8>
!i12i = !p4hir.int<12>
!i2i = !p4hir.int<2>
!i3i = !p4hir.int<3>
#true = #p4hir.bool<true> : !p4hir.bool
#int0_i2i = #p4hir.int<0> : !i2i
#int0_i3i = #p4hir.int<0> : !i3i
#int1_b8i = #p4hir.int<1> : !b8i
#int2_i12i = #p4hir.int<2> : !i12i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.func @funcA
  p4hir.func @funcA() -> tuple<!b8i, !i12i, !p4hir.bool> {
    // CHECK-DAG: %[[AGG_CST:.*]] = p4hir.const #p4hir.aggregate<[#int1_b8i, #int2_i12i, #true]> : tuple<!b8i, !i12i, !p4hir.bool>
    // CHECK-DAG: p4hir.soft_return %[[AGG_CST]] : tuple<!b8i, !i12i, !p4hir.bool>

    %c1_b8i = p4hir.const #int1_b8i
    %c2_i12i = p4hir.const #int2_i12i
    %true = p4hir.const #true
    %tuple = p4hir.tuple (%c1_b8i, %c2_i12i, %true) : tuple<!b8i, !i12i, !p4hir.bool>
    p4hir.soft_return %tuple : tuple<!b8i, !i12i, !p4hir.bool>
    p4hir.return
  }
  
  // CHECK-LABEL: p4hir.func @funcB
  p4hir.func @funcB() -> tuple<!b8i, !p4hir.bool, tuple<!i2i, !i3i>> {
    // CHECK-DAG: %[[AGG_CST:.*]] = p4hir.const #p4hir.aggregate<[#int1_b8i, #true, #p4hir.aggregate<[#int0_i2i, #int0_i3i]> : tuple<!i2i, !i3i>]> : tuple<!b8i, !p4hir.bool, tuple<!i2i, !i3i>>
    // CHECK-DAG: p4hir.soft_return %[[AGG_CST]] : tuple<!b8i, !p4hir.bool, tuple<!i2i, !i3i>>

    %c1_b8i = p4hir.const #int1_b8i
    %true = p4hir.const #true
    %c0_i2i = p4hir.const #int0_i2i
    %c0_i3i = p4hir.const #int0_i3i
    %tuple = p4hir.tuple (%c0_i2i, %c0_i3i) : tuple<!i2i, !i3i>
    %tuple_0 = p4hir.tuple (%c1_b8i, %true, %tuple) : tuple<!b8i, !p4hir.bool, tuple<!i2i, !i3i>>
    p4hir.soft_return %tuple_0 : tuple<!b8i, !p4hir.bool, tuple<!i2i, !i3i>>
    p4hir.return
  }
}

