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
// SerEnumElimination
//===----------------------------------------------------------------------===//

def SerEnumElimination : Pass<"p4hir-ser-enum-elimination", "mlir::ModuleOp"> {
  let summary = "Lower serializable enum types to their underlying types";
  let description = [{
    This pass converts all instances of `!p4hir.ser_enum` to the corresponding
    P4HIR::BitsType (`!p4hir.bit<N>`, or `!p4hir.int<N>`)
  }];

  let constructor = "P4MLIR::createSerEnumEliminationPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// Remove Aliases
//===----------------------------------------------------------------------===//

def RemoveAliases : Pass<"p4hir-remove-aliases", "mlir::ModuleOp"> {
  let summary = "Replace type aliases with their original types";
  let description = [{
    This pass eliminates all `!p4hir.alias` types by replacing them with their
    underlying aliased types. It recursively resolves nested aliases to their
    final concrete types.

    For example:
    ```mlir
    !b32i = !p4hir.bit<32>
    !T = !p4hir.alias<"T", !b32i>
    !Nested = !p4hir.alias<"Nested", !T>
    
    p4hir.func @process(%arg0: !Nested) {
      %c = p4hir.const #p4hir.int<10> : !Nested
      p4hir.return
    }
    ```

    After the pass:
    ```mlir
    !b32i = !p4hir.bit<32>
    
    p4hir.func @process(%arg0: !b32i) {
      %c = p4hir.const #p4hir.int<10> : !b32i
      p4hir.return
    }
    ```
  }];
  let constructor = "P4MLIR::createRemoveAliasesPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

def RemoveParserControlFlow : Pass<"p4hir-remove-parser-control-flow", "mlir::ModuleOp"> {
  let summary = "Replace control flow statements in parsers with transitions where possible";
  let description = [{
    Replace control flow statements by introducing new sub-states and appropriate transitions.

    For example, this code snippet:
    ```
    state s {
      statements1;
      if (cond)
          statements2;
      else
          statements3;
      statements4;
      transition selectExpression;
    }
    ```

    Would be converted to the equivalent of:
    ```
    state s {
      statements1;
      transition select(cond) {
          true: s_then;
          false: s_else;
      }
    }

    state s_then {
      statements2;
      transition s_join;
    }

    state s_else {
      statements3;
      transition s_join;
    }

    state s_join {
      statements4;
      transition selectExpression;
    }
    ```

    A similar transformation is applied for switch statements.
  }];
  let constructor = "P4MLIR::createRemoveParserControlFlowPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

#endif // P4MLIR_TRANSFORMS_PASSES_TD
