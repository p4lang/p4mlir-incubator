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
// Remove alias types
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

//===----------------------------------------------------------------------===//
// EnumElimination
//===----------------------------------------------------------------------===//

def EnumElimination : Pass<"p4hir-enum-elimination", "mlir::ModuleOp"> {
  let summary = "Lower enums to serializable enums";
  let description = [{
    This pass converts instances of `!p4hir.enum` to `!p4hir.ser_enum`.
    The underlying integer type and values for each enum are defined by
    `EnumRepresentationInterface`.
  }];

  let constructor = "P4MLIR::createEnumEliminationPass()";
}

//===----------------------------------------------------------------------===//
// ExpandEmit
//===----------------------------------------------------------------------===//

def ExpandEmit : Pass<"p4hir-expand-emit", "mlir::ModuleOp"> {
  let summary = "Expand struct emits";
  let description = [{
    This pass recursively converts `!p4corelib.emit` of `!p4hir.struct` to
    the sequence of emits of its fields.
  }];

  let constructor = "P4MLIR::createExpandEmitPass()";
  let dependentDialects = ["P4MLIR::P4CoreLib::P4CoreLibDialect"];
}

//===----------------------------------------------------------------------===//
// Inline Parsers
//===----------------------------------------------------------------------===//

def InlineParsers : Pass<"p4hir-inline-parsers", "mlir::ModuleOp"> {
  let summary = "Inline subparser calls";
  let description = [{
    This pass inlines subparser calls where possible.
    If successful, the subparser instantiation is replaced with the instantiation of
    its stateful objects and the states are inlined in the place of the apply call.
    The inlining will fail if the apply call is within control flow statements,
    as it won't be possible to split the corresponding state.
  }];
  let constructor = "P4MLIR::createInlineParsersPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// Inline Controls
//===----------------------------------------------------------------------===//

def InlineControls : Pass<"p4hir-inline-controls", "mlir::ModuleOp"> {
  let summary = "Inline sub-control calls";
  let description = [{
    This pass inlines sub-control calls where possible.
    If successful, the sub-control instantiation is replaced with the instantiation of
    its stateful objects, actions and tables are copied and the sub-control call is
    replaced with the inlined sub-control apply code.
  }];
  let constructor = "P4MLIR::createInlineControlsPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// Remove Soft Control Flow
//===----------------------------------------------------------------------===//

def RemoveSoftCF : Pass<"p4hir-remove-soft-cf"> {
  let summary = "Convert functions and controls to SESE form.";
  let description = [{
    Convert functions and controls to Single-Entry-Single-Exit (SESE) form by restructuring
    code and eliminating return, break and continue statements.

    For example code like this:
    ```
    bit<8> func(bit<8> a) {
      if (a > 200) { return 1; }

      if (a > 100) {
        a += 10;
      } else {
        return 2;
      }

      return a;
    }
    ```

    Will be transformed to something like this:
    ```
    bit<8> func(bit<8> a) {
      bit<8> result;
      if (a > 200) {
        result = 1;
      } else {
        if (a > 100) {
          a += 10;
          result = a;
        } else {
          result = 2;
        }
      }
      return result;
    }
    ```
  }];

  let constructor = "P4MLIR::createRemoveSoftCFPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// De-alias (aka copy-in/copy-out elimin
//===----------------------------------------------------------------------===//

def CopyInCopyOutElimination : Pass<"p4hir-copyin-copyout-elimination", "mlir::ModuleOp"> {
  let summary = "Remove extra call arguments copies due to copy-in/copy-out semantics";
  let description = [{
    P4 calls has copy-in / copy-out semantics for its arguments so we can assume
    that inout / out function parameters never alias. In order to enforce this
    copies for all inout arguments are created and inout/out arguments are read
    in very specific order. While semantically correct, this creates potentially
    inefficient code as additional local variable is created for every indirect call
    argument as well as corresponding loads and stores.

    This pass aims to match particular copy-in / copy-out argument passing patterns
    and remove extra variables ("de-alias") when we can prove that the callee
    accesses their arguments in compatible manner.

    E.g. the following code:
    ```mlir
    p4hir.parser @parserI(%arg0: !packet_in ..., %arg1: !p4hir.ref<!Parsed_packet> {p4hir.dir = #out, p4hir.param_name = "hdr"})() {
      p4hir.state @start {
        p4hir.scope {
          %ethernet_field_ref = p4hir.struct_field_ref %arg1["ethernet"] : <!Parsed_packet>
          %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!Ethernet_h>
          p4hir.call_method @packet_in::@extract<[!Ethernet_h]> of %arg0 : !packet_in (%hdr_out_arg) : (!p4hir.ref<!Ethernet_h>) -> ()
          %val_0 = p4hir.read %hdr_out_arg : <!Ethernet_h>
          p4hir.assign %val_0, %ethernet_field_ref : <!Ethernet_h>
        }
        ...
    ```
    would first be translated to (via P4CoreLib dialect):
    ```
    mlir
      p4hir.parser @parserI(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!Parsed_packet> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "hdr"})() {
      p4hir.state @start {
        p4hir.scope {
          %ethernet_field_ref = p4hir.struct_field_ref %arg1["ethernet"] : <!Parsed_packet>
          %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!Ethernet_h>
          p4corelib.extract_header %hdr_out_arg : <!Ethernet_h> from %arg0 : !p4corelib.packet_in
          %val_0 = p4hir.read %hdr_out_arg : <!Ethernet_h>
          p4hir.assign %val_0, %ethernet_field_ref : <!Ethernet_h>
        }
        ...
    ```
    and because `p4corelib.extract_header` has well-defined memory access semantics it could be turned into
    ```
    mlir
      p4hir.parser @parserI(%arg0: !p4corelib.packet_in {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "pkt"}, %arg1: !p4hir.ref<!Parsed_packet> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "hdr"})() {
      p4hir.state @start {
        p4hir.scope {
          %ethernet_field_ref = p4hir.struct_field_ref %arg1["ethernet"] : <!Parsed_packet>
          p4corelib.extract_header %%ethernet_field_ref : <!Ethernet_h> from %arg0 : !p4corelib.packet_in
        }
        ...
    ```
    eliminating variable as well as read / assign pair
  }];

  let constructor = "P4MLIR::createCopyInCopyOutEliminationPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// TupleToStruct
//===----------------------------------------------------------------------===//

def TupleToStruct : Pass<"p4hir-tuple-to-struct", "mlir::ModuleOp"> {
  let summary = "Convert tuple types/ops into p4hir.struct types/ops";
  let description = [{
    This pass converts all `tuple` types and operations into equivalent `p4hir.struct` types and operations.
  }];

  let constructor = "P4MLIR::createTupleToStructPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// SymbolDCE
//===----------------------------------------------------------------------===//

def SymbolDCE : Pass<"p4hir-symbol-dce"> {
  let summary = "Eliminate dead symbols";
  let description = [{
  }];
  let constructor = "P4MLIR::createSymbolDCEPass()";

  let statistics = [
    Statistic<"numDCE", "num-dce'd", "Number of symbols DCE'd">,
  ];
}

//===----------------------------------------------------------------------===//
// UseControlPlaneNames
//===----------------------------------------------------------------------===//

def UseControlPlaneNames : Pass<"p4hir-use-controlplane-names", "mlir::ModuleOp"> {
  let summary = "Renames Symbols to use their control plane names";
  let description = [{
    This pass renames some Symbols (tables etc) to use their control plane name.
  }];

  let constructor = "P4MLIR::createUseControlPlaneNamesPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

//===----------------------------------------------------------------------===//
// FlattenStructs
//===----------------------------------------------------------------------===//

def FlattenStructs : Pass<"p4hir-flatten-structs", "mlir::ModuleOp"> {
  let summary = "Flattens Structs";
  let description = [{
    This pass performs struct and header flattening, ensuring that all the struct types in
    the module contain only headers and scalars, and that headers contain only scalars.
  }];

  let constructor = "P4MLIR::createFlattenStructsPass()";
  let dependentDialects = ["P4MLIR::P4HIR::P4HIRDialect"];
}

#endif // P4MLIR_TRANSFORMS_PASSES_TD
