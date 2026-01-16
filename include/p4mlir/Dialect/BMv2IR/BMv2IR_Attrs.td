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

def BMv2IR_ActionTableAttr : BMv2IR_Attr<"ActionTable", "action_table"> {
  let summary = "Action-Table pair";
  let description = [{
    This attribute contains a pair of symbol references, one for an action and one for the
    corresponding next table in the next_table section of the BMv2 description of Tables.
    If the table symbol ref is not present, the current node is a terminating node in the pipeline.
  }];
  
  let parameters = (ins
    "mlir::SymbolRefAttr":$action,
    OptionalParameter<"mlir::SymbolRefAttr">:$table
  );
  
  let assemblyFormat = "`<` $action (`:` $table^)? `>`";
}

def BMv2IR_ActionTableArrayAttr : TypedArrayAttrBase<
  BMv2IR_ActionTableAttr, "array of action-table pair attributes">;

def BMv2IR_HitOrMissAttr : BMv2IR_Attr<"HitOrMiss", "hit_miss"> {
  let summary = "Next table for hit/miss checks";
  let description = [{
    This attribute holds a symbol reference for the next table in case of a hit, and a symbol reference
    for the miss. Both the symbol references are optional, but at least one of them should be present.
  }];
  
  let parameters = (ins
    OptionalParameter<"mlir::SymbolRefAttr">:$hitNode,
    OptionalParameter<"mlir::SymbolRefAttr">:$missNode
  );
  
  // TODO: add verifier, check that references are tables or conditionals and that at least one
  // of them is available.
  let assemblyFormat = "`<` (`hit` $hitNode^)? ` ` (`miss` $missNode^)? `>`";
}

def BMv2IR_TableMatchKind : IntEnum<"TableMatchKind", "BMv2IR table match kind", [
    I32EnumAttrCase<"Valid", 0, "valid">,
    I32EnumAttrCase<"Exact", 1, "exact">,
    I32EnumAttrCase<"LPM", 2, "lpm">,
    I32EnumAttrCase<"Ternary", 3, "ternary">,
    I32EnumAttrCase<"Range", 4, "range">
], 32> {
    let cppNamespace = "P4::P4MLIR::BMv2IR";
}

def BMv2IR_TableMatchKindAttr : EnumAttr<BMv2IR_Dialect, BMv2IR_TableMatchKind, "table_key_kind">;

def BMv2IR_TableKeyAttr : BMv2IR_Attr<"TableKey", "table_key"> {
  let summary = "Table key description attribute";
  let description = [{
    Represents one element in a table match key.
  }];

  let parameters = (ins
      "TableMatchKind":$match_type,
      "mlir::SymbolRefAttr":$header,
      OptionalParameter<"mlir::StringAttr">:$field_name,
      OptionalParameter<"mlir::IntegerAttr">:$mask,
      OptionalParameter<"mlir::StringAttr">:$name
  );

  // TODO: add verifier, symbol ref should be a header, field should be valid withing the header
  let assemblyFormat = [{
      `<` `type` $match_type `,` `header` $header (`[` $field_name^ `]`)? (`mask` `=` $mask^)? (`name` `=` $name^)? `>`
  }];
}

def BMv2IR_TableKeyArrayAttr : TypedArrayAttrBase<BMv2IR_TableKeyAttr,
    "Array of match key elements">;

def BMv2IR_TableDefaultEntryAttr : BMv2IR_Attr<"TableDefaultEntry", "table-default-entry"> {
  let summary = "Table default entry description attribute";
  let description = [{
    Represents the default entry of a table.
  }];

  let parameters = (ins
      "mlir::SymbolRefAttr":$action,
      DefaultValuedParameter<"bool", "false">:$action_const,
      OptionalArrayRefParameter<"std::string">:$action_data,
      DefaultValuedParameter<"bool", "false">:$action_entry_const
  );

  // TODO: add verifier: symbol ref should be an action, actionData should have as many entries as action args
  let assemblyFormat = [{
      `<` `action` $action `,` (`action_data`  `[` $action_data^ `]`)? `action_const` $action_const `,` `action_entry_const` $action_entry_const `>`
  }];
}

def BMv2IR_TableType : IntEnum<"TableType",
    "BMv2 Table Type",
    [
      I32EnumAttrCase<"Simple", 0, "simple">,
      I32EnumAttrCase<"Indirect", 1, "indirect">,
      I32EnumAttrCase<"IndirectWS", 2, "indirect_ws">
    ], 32> {
  let cppNamespace = "P4::P4MLIR::BMv2IR";
}

def BMv2IR_TableTypeAttr : EnumAttr<BMv2IR_Dialect, BMv2IR_TableType, "table_type">;

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_ATTRS_TD
