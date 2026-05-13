// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s | FileCheck %s

module {
  p4hir.func @log(!p4hir.string)
  // CHECK-LABEL: @test
  p4hir.func action @test() {
    // CHECK: %[[cst:.*]] = p4hir.const "This is a message" : !string
    %cst = p4hir.const "This is a message" : !p4hir.string
    // CHECK: p4hir.call @log (%[[cst]]) : (!string) -> ()
    p4hir.call @log (%cst) : (!p4hir.string) -> ()
    p4hir.return
  }
}
