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

//===----------------------------------------------------------------------===//
// SimplifyParsers
//===----------------------------------------------------------------------===//

def SimplifyParsers : Pass<"p4hir-simplify-parsers"> {
  let summary = "Simplifies parser control flow";
  let description = [{
    This pass removes parser states that are unreachable from the 'start' state
    and collapses linear chains of states into a single state.

    A linear chain is a sequence of states (e.g., s0 -> s1 -> s2) where each state
    has exactly one successor and each successor has exactly one predecessor
    with no annotations on any state in the chain.

    Special states ('start', 'accept', 'reject') are preserved.

    Example:

    Before:
      start -> s0 -> s1 -> accept

    After:
      start -> s0 -> accept
    (with operations from s1 merged into s0)
  }];
  let constructor = "P4MLIR::createSimplifyParsersPass()";
}

#endif // P4MLIR_TRANSFORMS_PASSES_TD
