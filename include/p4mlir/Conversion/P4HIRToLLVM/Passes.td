// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//
// This file contains definitions for P4HIR to LLVM dialect conversions
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_P4HIRTOLLVM_PASSES_TD
#define P4MLIR_CONVERSION_P4HIRTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerP4HIRToLLVM : Pass<"lower-p4hir-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers P4HIR to the LLVM dialect";
  let description = [{
    This pass lowers P4HIR operations and types to their LLVM dialect counterparts.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::mlir::LLVM::LLVMDialect",
  ];
}

#endif // P4MLIR_CONVERSION_P4HIRTOLLVM_PASSES_TD
