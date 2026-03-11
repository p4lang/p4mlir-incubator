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
    This pass lowers P4HIR operations and types to their BMv2IR counterparts.
    The pass assumes that Headers have already been lowered to HeaderInstances by
    `LowerToHeaderInstance`.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

def LowerToHeaderInstance : Pass<"lower-to-header-instance", "mlir::ModuleOp"> {
  let summary = "Lowers function arguments and local variables to header instances";
  let description = [{
    In BMv2 there is no concept of funcation arguments and local variables for header,
    everything lives in the same "global" space, the 
    [headers](https://github.com/p4lang/behavioral-model/blob/main/docs/JSON_format.md#headers)
    JSON node.
    To replicate this behaviour, this pass traverses the IR looking for function arguments and
    local variables whose type is either a header or a struct containing headers, it adds
    header_instance ops for it and replaces uses of the header (or the struct's fields) with the
    global instance.
    The pass only supports either:
    * headers with only bit fields
    * structs with only headers or bit fields
    If a struct has both bit fields and headers, the pass will create a new struct with only the
    bit fields (and a corresponding header instance), and separate instances for the header fields.
    Header instances are added at the start of the ModuleOp's block, and since operations may be
    `IsolatedFromAbove`, they are accessed via the SymToValue operation.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

#endif // P4MLIR_CONVERSION_P4HIRToBMv2IR_PASSES_TD
