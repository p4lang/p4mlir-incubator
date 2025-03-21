#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"

class P4HIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<P4HIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

def P4HIR_BoolAttr : P4HIR_Attr<"Bool", "bool", [TypedAttrInterface]> {
  let summary = "Represent true/false for !p4hir.bool types";
  let description = [{
    The BoolAttr represents a 'true' or 'false' value.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "mlir::Type">:$type,
                    "bool":$value);
  let builders = [
    AttrBuilder<(ins "bool":$value), [{
      return $_get($_ctxt, BoolType::get($_ctxt), value);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

//===----------------------------------------------------------------------===//
// IntAttr
//===----------------------------------------------------------------------===//

def P4HIR_IntAttr : P4HIR_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::APInt":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "const llvm::APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "int64_t":$value), [{
      BitsType intType = mlir::cast<BitsType>(type);
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];
  let extraClassDeclaration = [{
    int64_t getSInt() const { return getValue().getSExtValue(); }
    uint64_t getUInt() const { return getValue().getZExtValue(); }
    bool isNullValue() const { return getValue() == 0; }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// AggAttr
//===----------------------------------------------------------------------===//

def P4HIR_AggAttr : P4HIR_Attr<"Agg", "aggregate", [TypedAttrInterface]> {
  let summary = "An Attribute containing an aggregate value";
  let description = [{
    An aggregate attribute is a literal attribute that represents an aggregate
    value of the specified type. For nested aggregates, embedded arrays are
    used.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::ArrayAttr":$fields);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "mlir::ArrayAttr":$members), [{
      return $_get(type.getContext(), type, members);
    }]>
  ];
  // let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` $fields `>`
  }];

}

//===----------------------------------------------------------------------===//
// EnumFieldAttr
//===----------------------------------------------------------------------===//
// An attribute to indicate an enumeration value.
def P4HIR_EnumFieldAttr : P4HIR_Attr<"EnumField", "enum_field", [TypedAttrInterface]> {
 let summary = "Enumeration field attribute";
  let description = [{
    This attribute represents a field of an enumeration.

    Examples:
    ```mlir
      #p4hir.enum_field<A, !p4hir.enum<"name", A, B, C>> : !p4hir.enum<"name", A, B, C>
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::mlir::StringAttr":$field);

  // Force all clients to go through custom builder so we can check
  // whether the requested enum value is part of the provided enum type.
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "mlir::StringAttr": $value)>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "mlir::StringRef": $value), [{
      return $_get(type.getContext(), type, mlir::StringAttr::get(type.getContext(), value));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ErrorAttr
//===----------------------------------------------------------------------===//
// An attribute to indicate a particular error code.
// TODO: Decide if we'd want to unify with EnumFieldAttr?
def P4HIR_ErrorCodeAttr : P4HIR_Attr<"ErrorCode", "error", [TypedAttrInterface]> {
 let summary = "Error code attribute";
  let description = [{
    This attribute represents an error code.

    Examples:
    ```mlir
      #p4hir.error<ErrorA, !p4hir.error<ErrorA, ErrorB, ErrorC>> : !p4hir.error<ErrorA, ErrorB, ErrorC>
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::mlir::StringAttr":$field);

  // Force all clients to go through custom builder so we can check
  // whether the requested error value is part of the provided error type.
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "mlir::StringAttr": $value)>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "mlir::StringRef": $value), [{
      return $_get(type.getContext(), type, mlir::StringAttr::get(type.getContext(), value));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ValidAttr
//===----------------------------------------------------------------------===//

def ValidityBit_Invalid: I32BitEnumAttrCaseNone<"Invalid", "invalid">;
def ValidityBit_Valid  : I32BitEnumAttrCaseBit<"Valid", 0, "valid">;

def ValidityBit : I32BitEnumAttr<
    "ValidityBit",
    "validity of a header",
    [ValidityBit_Invalid, ValidityBit_Valid]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
  let genSpecializedAttr = 0;
}
def ValidityBitAttr : EnumAttr<P4HIR_Dialect, ValidityBit, "validity.bit",
                               [TypedAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::Type getType() { return P4HIR::ValidBitType::get(getContext()); }
  }];
}

//===----------------------------------------------------------------------===//
// ParamDirAttr
//===----------------------------------------------------------------------===//

def ParamDirection_None : I32EnumAttrCase<"None", 1, "undir">;
def ParamDirection_In   : I32EnumAttrCase<"In",   2, "in">;
def ParamDirection_Out  : I32EnumAttrCase<"Out",  3, "out">;
def ParamDirection_InOut: I32EnumAttrCase<"InOut",4, "inout">;

def ParamDirection : I32EnumAttr<
    "ParamDirection",
    "direction of parameter",
    [ParamDirection_None, ParamDirection_In,
     ParamDirection_Out, ParamDirection_InOut]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
  let genSpecializedAttr = 0;
}

def ParamDirAttr : EnumAttr<P4HIR_Dialect, ParamDirection, "dir">;

//===----------------------------------------------------------------------===//
// CtorParamAttr
//===----------------------------------------------------------------------===//

def P4HIR_CtorParamAttr : P4HIR_Attr<"CtorParam", "ctor_param", [TypedAttrInterface]> {
  let summary = "An Attribute containing an constructor parameter value";
  let description = [{
    An attribute to represent constructor argument value for parsers and controls.
    These values are compile-time constants but are not known until instantiation.
    The attribute serves as a placeholder that could be resolved during instantiation.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::SymbolRefAttr":$parent,
                        "mlir::StringAttr":$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "mlir::SymbolRefAttr":$parent,
                                        "mlir::StringAttr":$name), [{
      return $_get(type.getContext(), type, parent, name);
    }]>
  ];
  // let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` $parent `,` $name `>`
  }];

}

//===----------------------------------------------------------------------===//
// MatchKindAttr
//===----------------------------------------------------------------------===//
def P4HIR_MatchKindAttr : P4HIR_Attr<"MatchKind", "match_kind",
  [
   // TODO: Do we need this to be typed?
   // TypedAttrInterface
   ]> {
  let summary = "Represent match kind";
  let description = [{
    The MatchKindAttr represents a match kind value
  }];

  let parameters = (ins "mlir::StringAttr":$value);

  let assemblyFormat = [{
    `<` $value `>`
  }];

  let builders = [
    AttrBuilder<(ins "mlir::StringRef":$value), [{
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, value));
    }]>
  ];

}

//===----------------------------------------------------------------------===//
// CaseOpKind
//===----------------------------------------------------------------------===//

def CaseOpKind_DT : I32EnumAttrCase<"Default", 1, "default">;
def CaseOpKind_EQ : I32EnumAttrCase<"Equal", 2, "equal">;
def CaseOpKind_AO : I32EnumAttrCase<"Anyof", 3, "anyof">;

def CaseOpKind : I32EnumAttr<
    "CaseOpKind",
    "case kind",
    [CaseOpKind_DT, CaseOpKind_EQ, CaseOpKind_AO]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}


#endif // P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_TD
