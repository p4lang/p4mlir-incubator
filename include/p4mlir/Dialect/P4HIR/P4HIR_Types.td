#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"

//===----------------------------------------------------------------------===//
// P4HIR type definitions.
//===----------------------------------------------------------------------===//

class P4HIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4HIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def P4HIR_BitType : P4HIR_Type<"BitType", "bit"> {
  let summary = "fixed width unsigned integer type";
  let description = [{
    The `p4hir.bit` type represents a P4 `bit` type.

    For example, `p4hir.bit<32>` represents `bit<32>` in P4.
  }];
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
}

//===----------------------------------------------------------------------===//
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def P4HIR_P4Type : AnyTypeOf<[P4HIR_BitType]> {}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
