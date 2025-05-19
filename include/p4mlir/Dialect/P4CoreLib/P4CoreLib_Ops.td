#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD

include "mlir/IR/OpBase.td"

include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

//===----------------------------------------------------------------------===//
// Base P4CoreLob operation definition.
//===----------------------------------------------------------------------===//

class P4CoreLib_Op<string mnemonic, list<Trait> traits = []> :
        Op<P4CoreLib_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// P4CoreLib operation definitions.
//===----------------------------------------------------------------------===//

def VerifyOp : P4CoreLib_Op<"verify",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>]> {
    let summary = "Defines a P4 constant";
    let description = [{
        Check a predicate @check in the parser; if the predicate is true do nothing,
        otherwise set the parser error to @toSignal, and transition to the `reject` state.
    }];

    let arguments = (ins BooleanType:$check, ErrorType:$toSignal);
    let assemblyFormat = [{
      $check `signalling` $toSignal `:` type($toSignal) attr-dict
    }];
}


#endif //P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD
