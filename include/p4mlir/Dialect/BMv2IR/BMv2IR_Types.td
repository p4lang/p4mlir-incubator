#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"

//===----------------------------------------------------------------------===//
// BMv2IR type definitions.
//===----------------------------------------------------------------------===//

class BMv2IR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BMv2IR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD
