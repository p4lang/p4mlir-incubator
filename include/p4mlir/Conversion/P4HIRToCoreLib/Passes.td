//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_PASSES_TD
#define P4MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerToP4CoreLib : Pass<"lower-to-p4corelib", "mlir::ModuleOp"> {
  let summary = "Convert P4HIR extern calls to P4 corelib ops";
  let description = [{
    This pass converts explicit P4HIR extern calls into P4 corelib operations
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
  ];
}

#endif // P4MLIR_CONVERSION_PASSES_TD
