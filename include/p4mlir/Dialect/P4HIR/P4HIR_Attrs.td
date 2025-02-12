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

  let parameters = (ins AttributeSelfTypeParameter<"", "BoolType">:$type,
                    "bool":$value);

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
  //let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    `<` $fields `>`
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


#endif // P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_TD
