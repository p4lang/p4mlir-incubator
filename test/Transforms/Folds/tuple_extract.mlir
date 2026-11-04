// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!b10i = !p4hir.bit<10>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
#int0_b32i = #p4hir.int<0> : !b32i
#int10_b10i = #p4hir.int<10> : !b10i
#int10_b32i = #p4hir.int<10> : !b32i
#int12_b16i = #p4hir.int<12> : !b16i
#int7_b8i = #p4hir.int<7> : !b8i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: p4hir.func @func1
  p4hir.func @func1(%arg0: !b8i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "a"}) -> !b8i {
    // CHECK-NEXT: p4hir.soft_return %arg0 : !b8i

    %c10_b10i = p4hir.const #int10_b10i
    %tuple = p4hir.tuple(%c10_b10i, %arg0) : tuple<!b10i, !b8i>
    %t1 = p4hir.tuple_extract %tuple[1] : tuple<!b10i, !b8i>
    p4hir.soft_return %t1 : !b8i
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @func2
  p4hir.func @func2() -> !b8i {
    // CHECK-NEXT: %c7_b8i = p4hir.const #int7_b8i
    // CHECK-NEXT: p4hir.soft_return %c7_b8i : !b8i

    %cst = p4hir.const #p4hir.aggregate<[#int10_b10i, #int7_b8i]> : tuple<!b10i, !b8i>
    %t1 = p4hir.tuple_extract %cst[1] : tuple<!b10i, !b8i>
    p4hir.soft_return %t1 : !b8i
    p4hir.return
  }
}

