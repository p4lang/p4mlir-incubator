#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

//===----------------------------------------------------------------------===//
// Base P4HIR operation definition.
//===----------------------------------------------------------------------===//

class P4HIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<P4HIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// P4HIR operation definitions.
//===----------------------------------------------------------------------===//

def P4HIR_ConstOp : P4HIR_Op<"const", [Pure]> {
    let summary = "const operation";
    let description = [{
        The `p4hir.const` operation turns a literal into an SSA value. It
        represents a constant declaration in P4.

        Example:

        ```mlir
        %0 = p4hir.const 42 : ui32 -> !p4hir.bit<32>
        ```

        Corresponding P4 code:

        ```p4
        const bit<32> a = 42;
        ```
    }];

    let arguments = (ins APIntAttr:$value);
    let results = (outs P4HIR_P4Type:$result);

    let assemblyFormat = [{
        $value attr-dict `->` qualified(type($result))
    }];
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
