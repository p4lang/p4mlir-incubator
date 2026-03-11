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

def LowerPackage : Pass<"lower-package", "mlir::ModuleOp"> {
  let summary = "Lowers P4HIR Package instantiations";
  let description = [{
    This pass lowers P4HIR package instantiation to target-specific ops.
    It also ensure that any target-specific requirement about naming parsers and controls is satisfied.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

def SynthesizeActions : Pass<"synth-actions", "mlir::ModuleOp"> {
  let summary = "Synthesizes dummy actions for raw operations in control_apply blocks";
  let description = [{
    BMv2 doesn't have a way of expressing arbitrary operations in control_apply blocks:
    it can only express either table_apply or control flow nodes. If a control_apply
    block contains a "raw" [sequence of] operations, we need to enclose the operation(s)
    in a dummy action and call that action from the control_apply block.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

def SynthesizeTables : Pass<"synth-tables", "mlir::ModuleOp"> {
  let summary = "Synthesizes dummy tables for action calls in control_apply blocks";
  let description = [{
    Synthesizes dummy tables that wrap action calls in control_apply blocks.
    Note that technically BMv2 could express action calls in control_apply, but the default behaviour
    in p4c is to always wrap them in tables (see Note 1 in the
    [Pipelines section](https://github.com/p4lang/behavioral-model/blob/main/docs/JSON_format.md#pipelines))
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
  ];
}

def SetCorelib : Pass<"set-corelib", "mlir::ModuleOp"> {
  let summary = "Identifies Corelib Ops";
  let description = [{
    Identifies Corelib operations and sets the `corelib` annotation, preparing for the lowering to
    P4CoreLib.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
  ];
}

def EnsureStandardMetadata : Pass<"ensure-standard-metadata", "mlir::ModuleOp"> {
  let summary = "Ensures that the standard_metadata header instance is in the Module";
  let description = [{
    Some BMv2 targets require the standard_metadata header instance to be in the module even if it's never
    used. This pass ensures this.
  }];
  let dependentDialects = [
      "::P4::P4MLIR::BMv2IR::BMv2IRDialect",
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
  ];
}

#endif // P4MLIR_CONVERSION_P4HIRToBMv2IR_PASSES_TD
