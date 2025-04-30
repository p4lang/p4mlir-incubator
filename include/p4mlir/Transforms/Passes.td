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

//===----------------------------------------------------------------------===//
// SimplifyParsers
//===----------------------------------------------------------------------===//

def SimplifyParsers : Pass<"p4hir-simplify-parsers"> {
  let summary = "Simplifies parser control flow";
  let description = [{
    This pass removes parser states that are unreachable from the 'start' state
    and collapses linear chains of states into a single state.

    Linear chains are any two states where s1 has exactly one outgoing edge
    to s2 and s2 has exactly one incoming edge from s1 with no annotations
    in any state in the chain.

    Example:

    Before:
      start --> s0 ---> s1 ---> s3 ---> s4 ---> accept
                |               ^
                v               |
                s2--------------+

    After:
      start ---> s1 ---> s3 (with accept terminator)
         |               ^
         v               |
         s2--------------+

    where:
      start = start + s0
      s3 = s3 + s4 + accept
  }];
  let constructor = "P4MLIR::createSimplifyParsersPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

#endif // P4MLIR_TRANSFORMS_PASSES_TD
