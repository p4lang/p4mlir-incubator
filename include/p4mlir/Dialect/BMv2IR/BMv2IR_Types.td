#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

//===----------------------------------------------------------------------===//
// BMv2IR type definitions.
//===----------------------------------------------------------------------===//

class BMv2IR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BMv2IR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// We use a custom parameter here to allow for an empty array of fields
def FieldInfoArrayParameter : ArrayRefParameter<"BMv2IR::FieldInfo", "struct fields"> {
  let parser = "::mlir::FieldParser<::llvm::ArrayRef<BMv2IR::FieldInfo>>::parse($_parser)";
  let printer = [{
     $_printer << '[';
     llvm::interleaveComma($_self, $_printer, [&](BMv2IR::FieldInfo field) {
         $_printer << field;
     });
     $_printer << ']';
   }];
}

def BMv2IR_HeaderType : BMv2IR_Type<"Header", "header"> {
  let summary = "BMv2 header type";
  let description = [{
    Represents a P4 header according to the
    [BMv2 JSON spec](https://github.com/p4lang/behavioral-model/blob/main/docs/JSON_format.md#header_types)
    
    The type encodes:
    - A unique name for the header type
    - An array of fields (name, bitwidth, optional signed flag)
    - Optional maximum length for variable-length headers
    
    At most one field can be variable-length.
  }];
  
  let parameters = (ins
    StringRefParameter<"struct name">:$name,
    FieldInfoArrayParameter:$fields,
    OptionalParameter<"unsigned">:$max_length
  );
  
  let assemblyFormat = [{
    `<` $name `,` $fields
    (`,` `max_length` `=` $max_length^)?
    `>`
  }];
  
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
     TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<BMv2IR::FieldInfo>":$fields), [{
      return $_get($_ctxt, name, fields, computeMaxLength(fields));
    }]>,
  ];
  
  // Add verification for constraints
  let extraClassDeclaration = [{
    static unsigned computeMaxLength(llvm::ArrayRef<BMv2IR::FieldInfo>);
    static bool isAllowedFieldType(mlir::Type ty);
    static constexpr llvm::StringRef validBitFieldName = "$valid$";
    bool hasField(llvm::StringRef name);
    llvm::FailureOr<BMv2IR::FieldInfo> getField(llvm::StringRef name);
  }];
}

def BMv2IR_FieldType : AnyTypeOf<[BitsType, VarBitsType]>;

def BMv2IR_HeaderUnionType : BMv2IR_Type<"HeaderUnion", "header_union"> {
  let summary = "BMv2 header union type";
  let description = [{
    Represents a BMv2 Header Union.
  }];
  
  let parameters = (ins
    StringRefParameter<"struct name">:$name,
    ArrayRefParameter<"BMv2IR::FieldInfo", "struct fields">:$fields
  );
  
  let assemblyFormat = [{
    `<` $name `,` `[` $fields `]` `>`
  }];
  
  let genVerifyDecl = 1;
}

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_TYPES_TD
