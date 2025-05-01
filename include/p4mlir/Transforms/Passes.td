#ifndef P4MLIR_TRANSFORMS_PASSES_TD
#define P4MLIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"

def PrintParsersGraph : Pass<"p4hir-print-parsers-graph"> {
  let summary = "Print parsers graph";
  let description = [{
    Print a parser (states and transitions) as a .dot graph
  }];

  let constructor = "P4MLIR::createPrintParsersGraphPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

#endif // P4MLIR_TRANSFORMS_PASSES_TD
