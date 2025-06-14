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

//===----------------------------------------------------------------------===//
// FlattenCFG
//===----------------------------------------------------------------------===//

def FlattenCFG : Pass<"p4hir-flatten-cfg"> {
  let summary = "Produces flatten CFG";
  let description = [{
    This pass transforms P4HIR and inline all the nested regions. Thus,
    the next post condtions are met after the pass applied:
    - there is no any nested region in a function / action / parser body (same for control.apply region)
    - all the blocks in a function belong to the parent region
    In other words, this pass removes such P4HIR operations like IfOp, ForOp,
    ForInOp, ScopeOp and etc. and produces a flat P4HIR.

    For example, the following code:
    ```mlir
      %false = p4hir.const #p4hir.bool<false> : !p4hir.bool
      p4hir.if %false {
        %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
      }
      p4hir.return
    ```
    is transformed into:
    ```mlir
      %false = p4hir.const #p4hir.bool<false> : !p4hir.bool
       p4hir.cond_br %false ^bb1, ^bb2
       ^bb1:
         %true = p4hir.const #p4hir.bool<true> : !p4hir.bool
         p4hir.br ^bb2
       ^bb2:
         p4hir.return
    ```
  }];
  let constructor = "P4MLIR::createFlattenCFGPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// RemoveParserControlFlow
//===----------------------------------------------------------------------===//

def RemoveParserControlFlow : Pass<"p4hir-remove-parser-control-flow"> {
  let summary = "Removes the control flow";
  let description = [{
    The pass transforms the if ops into switch ops.
    for example, the following code:
    ```mlir
      %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
      p4hir.if %0 {
        %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
      } else {
        %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
      }
    ```
    is transformed into:
    ```mlir
      %cond = p4hir.const #p4hir.bool<false> : !p4hir.bool
      %int_cond = p4hir.sext %cond : !p4hir.bool to !b32i
      #true_val = #p4hir.int<1> : !b32i
      #false_val = #p4hir.int<0> : !b32i
      p4hir.switch (%int_cond : !b32i) {
        p4hir.case(equal, [#true_val]) {
          %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
          p4hir.yield
        }
        p4hir.case(equal, [#false_val]) {
          %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
          p4hir.yield
        }
        p4hir.yield
      }
    ```
  }];
  let constructor = "P4MLIR::createRemoveParserControlFlowPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}
#endif // P4MLIR_TRANSFORMS_PASSES_TD
