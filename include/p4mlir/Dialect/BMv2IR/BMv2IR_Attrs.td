#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"

class BMv2IR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<BMv2IR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD
