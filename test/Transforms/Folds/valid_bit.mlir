// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

// MLIR for:
// header h { bit<8> a; }
// bool func1(in h h1) { return h1.isValid() == true; }
// bool func2(in h h1, in h h2) { return h1.isValid() != !h2.isValid(); }
// bool func3(in h h1) { return h1.isValid() != (h1 == (h) {#}); }

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
#true = #p4hir.bool<true> : !p4hir.bool
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
#invalid = #p4hir<validity.bit invalid> : !validity_bit
#valid = #p4hir<validity.bit valid> : !validity_bit
module @p4_main {
  // CHECK-LABEL: p4hir.func @func1
  p4hir.func @func1(%arg0: !h {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h1"}) -> !p4hir.bool {
    // CHECK-DAG: %[[V1:.*]] = p4hir.struct_extract %arg0["__valid"] : !h
    // CHECK-DAG: %[[EQ:.*]] = p4hir.cmp(eq, %[[V1]] : !validity_bit, %valid : !validity_bit)
    // CHECK-DAG: p4hir.soft_return %[[EQ]] : !p4hir.bool

    %__valid = p4hir.struct_extract %arg0["__valid"] : !h
    %valid = p4hir.const #valid
    %eq = p4hir.cmp(eq, %__valid : !validity_bit, %valid : !validity_bit)
    %true = p4hir.const #true
    %eq_0 = p4hir.cmp(eq, %eq : !p4hir.bool, %true : !p4hir.bool)
    p4hir.soft_return %eq_0 : !p4hir.bool
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @func2
  p4hir.func @func2(%arg0: !h {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h1"}, %arg1: !h {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h2"}) -> !p4hir.bool {
    // CHECK-DAG: %[[V1:.*]] = p4hir.struct_extract %arg0["__valid"] : !h
    // CHECK-DAG: %[[V2:.*]] = p4hir.struct_extract %arg1["__valid"] : !h
    // CHECK-DAG: %[[EQ:.*]] = p4hir.cmp(eq, %[[V1]] : !validity_bit, %[[V2]] : !validity_bit)
    // CHECK-DAG: p4hir.soft_return %[[EQ]] : !p4hir.bool

    %__valid = p4hir.struct_extract %arg0["__valid"] : !h
    %valid = p4hir.const #valid
    %eq = p4hir.cmp(eq, %__valid : !validity_bit, %valid : !validity_bit)
    %__valid_0 = p4hir.struct_extract %arg1["__valid"] : !h
    %valid_1 = p4hir.const #valid
    %eq_2 = p4hir.cmp(eq, %__valid_0 : !validity_bit, %valid_1 : !validity_bit)
    %not = p4hir.unary(not, %eq_2) : !p4hir.bool
    %ne = p4hir.cmp(ne, %eq : !p4hir.bool, %not : !p4hir.bool)
    p4hir.soft_return %ne : !p4hir.bool
    p4hir.return
  }

  // CHECK-LABEL: p4hir.func @func3
  p4hir.func @func3(%arg0: !h {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "h1"}) -> !p4hir.bool {
    // CHECK-DAG: %[[V1:.*]] = p4hir.struct_extract %arg0["__valid"] : !h
    // CHECK-DAG: %[[V2:.*]] = p4hir.struct_extract %arg0["__valid"] : !h
    // CHECK-DAG: %[[EQ:.*]] = p4hir.cmp(eq, %[[V1]] : !validity_bit, %[[V2]] : !validity_bit)
    // CHECK-DAG: p4hir.soft_return %[[EQ]] : !p4hir.bool

    %__valid = p4hir.struct_extract %arg0["__valid"] : !h
    %valid = p4hir.const #valid
    %eq = p4hir.cmp(eq, %__valid : !validity_bit, %valid : !validity_bit)
    %__valid_0 = p4hir.struct_extract %arg0["__valid"] : !h
    %invalid = p4hir.const #invalid
    %eq_1 = p4hir.cmp(eq, %__valid_0 : !validity_bit, %invalid : !validity_bit)
    %ne = p4hir.cmp(ne, %eq : !p4hir.bool, %eq_1 : !p4hir.bool)
    p4hir.soft_return %ne : !p4hir.bool
    p4hir.return
  }
}
