//===----------------------------------------------------------------------===//
//
// This file contains definitions for P4HIR to BMv2IR conversions
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_P4HIRToBMv2IR_PASSES_TD
#define P4MLIR_CONVERSION_P4HIRToBMv2IR_PASSES_TD

include "mlir/Pass/PassBase.td"

def P4HIRToBmv2IR : Pass<"p4hir-to-bmv2ir", "mlir::ModuleOp"> {
  let summary = "Lowers P4HIR to BMv2IR";
  let description = [{
    This pass lowers P4HIR operations and types to their BMv2IR counterparts
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

#endif // P4MLIR_CONVERSION_P4HIRToBMv2IR_PASSES_TD
