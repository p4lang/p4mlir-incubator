#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"

class BMv2IR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<BMv2IR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def BMv2IR_ExtractKind : IntEnum<"ExtractKind",
    "BMv2 extract operation kind",
    [
      I32EnumAttrCase<"Regular", 0, "regular">,
      I32EnumAttrCase<"Stack", 1, "stack">,
      I32EnumAttrCase<"UnionStack", 2, "union_stack">
    ], 32> {
  let cppNamespace = "P4::P4MLIR::BMv2IR";
}

def BMv2IR_ExtractKindAttr : EnumAttr<BMv2IR_Dialect, BMv2IR_ExtractKind, "extract_kind">;

def BMv2IR_TransitionKind : IntEnum<"TransitionKind",
    "BMv2 extract operation kind",
    [
      I32EnumAttrCase<"Default", 0, "default">,
      I32EnumAttrCase<"Hexstr", 1, "hexstr">,
      I32EnumAttrCase<"Parse_vset", 2, "parse_vset">
    ], 32> {
  let cppNamespace = "P4::P4MLIR::BMv2IR";
}

def BMv2IR_TransitionKindAttr : EnumAttr<BMv2IR_Dialect, BMv2IR_TransitionKind, "transition_kind">;

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD
