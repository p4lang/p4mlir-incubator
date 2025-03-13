#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.td"

//===----------------------------------------------------------------------===//
// P4HIR type definitions.
//===----------------------------------------------------------------------===//

class P4HIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4HIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class P4HIR_TypeNoMnemonic<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4HIR_Dialect, name, traits> {
    string typeName = dialect.name # "." # typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Integer types: signed (int<>), unsigned (bit<>) and arbitrary width (int)
//===----------------------------------------------------------------------===//

def BitsType : P4HIR_TypeNoMnemonic<"Bits", "bits"> {
  let summary = "fixed width integer type";
  let description = [{
    The `p4hir.bit` / `p4hir.int` type represents a P4 `bit` or `int` type.

    For example, `p4hir.bit<32>` represents `bit<32>` in P4 and
    `p4hir.int<42>` is `int<42>` in P4.
  }];
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let extraClassDeclaration = [{
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getIsSigned(); }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return !getIsSigned(); }

    static mlir::Type parse(mlir::AsmParser &parser, bool isSigned);
    void print(mlir::AsmPrinter &printer) const;

    std::string getAlias() const {
      return (isSigned() ? 'i' : 'b') + std::to_string(getWidth()) + 'i';
    };
  }];
}

// Unsigned integer type of a specific width (bits<width>).
class Bit<int width>
  : Type<And<[
        CPred<"::mlir::isa<::p4hir::BitsType>($_self)">,
        CPred<"::mlir::cast<::p4hir::BitsType>($_self).isUnsigned()">,
        CPred<"::mlir::cast<::p4hir::BitsType>($_self).getWidth() == " # width>
        ]>, width # "-bit unsigned integer", "::p4hir::BitsType">,
    BuildableType<
      "p4hir::BitsType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

// Signed integer type of a specific width (int<width>).
class Int<int width>
  : Type<And<[
        CPred<"::mlir::isa<::p4hir::IntType>($_self)">,
        CPred<"::mlir::cast<::p4hir::IntType>($_self).isSigned()">,
        CPred<"::mlir::cast<::p4hir::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit signed integer", "::p4hir::BitsType">,
    BuildableType<
      "p4hir::BitsType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

def InfIntType : P4HIR_Type<"InfInt", "infint"> {
  let summary = "arbitrary precision integer type";
  let description = [{
    `p4hir.infint` represents a P4 `int` type.
  }];

  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "infint"; };
  }];

}

//===----------------------------------------------------------------------===//
// BooleanType: An alternative here is to represent bool as mlir::i1, but
// let's be more generic, it could be lowered later on
//===----------------------------------------------------------------------===//

def BooleanType : P4HIR_Type<"Bool", "bool"> {
  let summary = "boolean type";
  let description = [{
    `p4hir.bool` represents a P4 `bool` type.
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "bool"; };
  }];

}

//===----------------------------------------------------------------------===//
// "Singleton" types
//===----------------------------------------------------------------------===//

def DontcareType : P4HIR_Type<"Dontcare", "dontcare"> {}
def UnknownType : P4HIR_Type<"Unknown", "unknown"> {}

def VoidType : P4HIR_Type<"Void", "void"> {
  let summary = "void type";
  let description = [{
    Represents absense of result of actions and methods, or `void` type for functions.
  }];
  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "void"; };
  }];
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def StringType : P4HIR_Type<"String", "string"> {
  let summary = "string type";
  let description = [{
    `p4hir.string` represents a P4 `string` type used for string literals.
  }];

  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "string"; };
  }];
}

//===----------------------------------------------------------------------===//
// ReferenceType
//===----------------------------------------------------------------------===//

def ReferenceType : P4HIR_Type<"Reference", "ref"> {
  let summary = "reference type";
  let description = [{
    `p4hir.ref` is a type returned by any op referring to an object in P4.
    This is an implementation detail of P4HIR dialect and should be used in
    very limited cases (e.g. `p4hir.alloca` result type is a reference type).
  }];

  let parameters = (ins "mlir::Type":$objectType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$objectType), [{
      return $_get(objectType.getContext(), objectType);
    }]>,
    TypeBuilder<(ins "mlir::Type":$objectType), [{
      return $_get($_ctxt, objectType);
    }]>
  ];

  let assemblyFormat = [{
    `<` $objectType `>`
  }];

  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def FuncType : P4HIR_Type<"Func", "func"> {
  let summary = "P4 function-like type (actions, methods, functions)";
  let description = [{
    The `!p4hir.func` is a function type.
    Example:

    ```mlir
    !p4hir.func<()>
    !p4hir.func<!p4hir.bit<32>(!p4hir.bit<32>, !p4hir.int<42>)>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$inputs,
                        "mlir::Type":$optionalReturnType,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments);

  let builders = [
    // Construct with an actual return type or explicit !p4hir.void
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$inputs, "mlir::Type":$returnType), [{
      return $_get(returnType.getContext(), inputs,
                   mlir::isa<P4HIR::VoidType>(returnType) ? nullptr
                                                          : returnType,
                   llvm::ArrayRef<mlir::Type>());
    }]>,

    // Construct with an actual return type or explicit !p4hir.void
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$inputs, "mlir::Type":$returnType, "llvm::ArrayRef<mlir::Type>":$typeArgs), [{
      return $_get(returnType.getContext(), inputs,
                   mlir::isa<P4HIR::VoidType>(returnType) ? nullptr
                                                          : returnType,
                   typeArgs);
    }]>,

    // Construct without return type
    TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$inputs), [{
      return $_get($_ctxt, inputs, nullptr, llvm::ArrayRef<mlir::Type>());
    }]>

  ];

  // Use a custom parser to handle the argument types in better way.
  let assemblyFormat = [{
    `<` (`<` $type_arguments^ `>`)? custom<FuncType>($inputs, $optionalReturnType) `>`
  }];

  let extraClassDeclaration = [{
    /// Returns the `i`th input operand type. Asserts if out of bounds.
    mlir::Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Returns the result type of the function as an actual return type or
    /// explicit !p4hir.void
    mlir::Type getReturnType() const;

    /// Returns the result type of the function as an ArrayRef, enabling better
    /// integration with generic MLIR utilities.
    llvm::ArrayRef<mlir::Type> getReturnTypes() const;

    /// Returns a clone of this action type with the given argument
    /// and result types. Required for FunctionOp interface
    FuncType clone(mlir::TypeRange inputs, mlir::TypeRange outputs) const;

    /// Returns whether the function returns void.
    bool isVoid() const;
  }];
}

//===----------------------------------------------------------------------===//
// ParserType
//===----------------------------------------------------------------------===//
def ParserType : P4HIR_Type<"Parser", "parser"> {
  let parameters = (ins StringRefParameter<"parser type name">:$name,
                        ArrayRefParameter<"mlir::Type">:$inputs);
  let assemblyFormat = [{
    `<` $name `,` custom<FuncType>($inputs) `>`
  }];
}

//===----------------------------------------------------------------------===//
// CtorType
//===----------------------------------------------------------------------===//
// We keep parameter names to simplify matching for ctor arguments during
// instantiation and specialization.
def CtorType : P4HIR_Type<"Ctor", "ctor"> {
  let parameters = (ins ArrayRefParameter<"std::pair<mlir::StringAttr, mlir::Type>">:$inputs,
                        "mlir::Type":$returnType);

  let builders = [
    // Construct with an actual return type
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>>":$inputs, "mlir::Type":$returnType), [{
      return $_get(returnType.getContext(), inputs, returnType);
    }]>
  ];

  let assemblyFormat = [{
    `<` custom<CtorType>($inputs, $returnType) `>`
  }];

  let extraClassDeclaration = [{
    /// Returns the `i`th input operand type. Asserts if out of bounds.
    mlir::Type getInput(unsigned i) const { return getInputs()[i].second; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

class StructLikeType<string name, string typeMnemonic>
  : P4HIR_Type<name, typeMnemonic,  [
  StructLikeTypeInterface,
  DestructurableTypeInterface,
  FieldIDTypeInterface
]> {
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let parameters = (
    ins StringRefParameter<"struct name">:$name,
        ArrayRefParameter<"P4HIR::FieldInfo", "struct fields">:$elements
  );

  string extra2ClassDeclaration = "";
  let extraClassDeclaration = !strconcat([{
    mlir::Type getFieldType(mlir::StringRef fieldName) {
      for (const auto &field : getElements())
        if (field.name == fieldName) return field.type;
       return {};
    }

    std::optional<P4HIR::FieldInfo> getField(mlir::StringRef fieldName) {
      for (const auto &field : getElements())
        if (field.name == fieldName) return field;
       return {};
    }

    llvm::ArrayRef<P4HIR::FieldInfo> getFields() const {
      return getElements();
    }

    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type> &types) {
      for (const auto &field : getElements()) types.push_back(field.type);
    }
    
    std::optional<unsigned> getFieldIndex(mlir::StringRef fieldName) {
      llvm::ArrayRef<P4HIR::FieldInfo> elems = getElements();
      for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
        if (elems[idx].name == fieldName) return idx;
      return {};
    }
    
    std::optional<unsigned> getFieldIndex(mlir::StringAttr fieldName) {
      llvm::ArrayRef<P4HIR::FieldInfo> elems = getElements();
      for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
        if (elems[idx].name == fieldName) return idx;
      return {};
    }

    // FieldID type interface implementation
    std::pair<unsigned, llvm::SmallVector<unsigned>> getFieldIDsStruct() const {
      unsigned fieldID = 0;
      auto elements = getElements();
      llvm::SmallVector<unsigned> fieldIDs;
      fieldIDs.reserve(elements.size());
      for (auto &element : elements) {
          auto type = element.type;
          fieldID += 1;
          fieldIDs.push_back(fieldID);
          // Increment the field ID for the next field by the number of subfields.
          fieldID += FieldIdImpl::getMaxFieldID(type);
      }
      return {fieldID, fieldIDs};
    }

    std::pair<mlir::Type, unsigned> getSubTypeByFieldID(unsigned fieldID) const {
      if (fieldID == 0) return {*this, 0};
      auto [maxId, fieldIDs] = getFieldIDsStruct();
      auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
      auto subfieldIndex = std::distance(fieldIDs.begin(), it);
      auto subfieldType = getElements()[subfieldIndex].type;
      auto subfieldID = fieldID - fieldIDs[subfieldIndex];
      return {subfieldType, subfieldID};
    }

    mlir::Type getTypeAtIndex(mlir::Attribute index) const {
      auto indexAttr = mlir::dyn_cast<mlir::IntegerAttr>(index);
      if (!indexAttr) return {};

      return getSubTypeByFieldID(indexAttr.getInt()).first;
    }

    unsigned getFieldID(unsigned index) const {
      auto [maxId, fieldIDs] = getFieldIDsStruct();
      return fieldIDs[index];
    }

    unsigned getMaxFieldID() const {
      unsigned fieldID = 0;
      for (const auto &field : getElements()) fieldID += 1 + FieldIdImpl::getMaxFieldID(field.type);
      return fieldID;
    }

    unsigned getIndexForFieldID(unsigned fieldID) const {
      assert(!getElements().empty() && "struct must have >0 fields");
      auto [maxId, fieldIDs] = getFieldIDsStruct();
      auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
      return std::distance(fieldIDs.begin(), it);
    }

    std::pair<unsigned, unsigned> getIndexAndSubfieldID(unsigned fieldID) const {
      auto index = getIndexForFieldID(fieldID);
      auto elementFieldID = getFieldID(index);
      return {index, fieldID - elementFieldID};
    }

    std::pair<unsigned, bool> projectToChildFieldID(unsigned fieldID,
                                                    unsigned index) const {
      auto [maxId, fieldIDs] = getFieldIDsStruct();
      auto childRoot = fieldIDs[index];
      auto rangeEnd = index + 1 >= getElements().size() ? maxId : (fieldIDs[index + 1] - 1);
      return std::make_pair(fieldID - childRoot, fieldID >= childRoot && fieldID <= rangeEnd);
    }

    std::optional<llvm::DenseMap<mlir::Attribute, mlir::Type>> getSubelementIndexMap() const {
      llvm::DenseMap<mlir::Attribute, mlir::Type> destructured;
      for (auto [i, field] : llvm::enumerate(getElements()))
          destructured.try_emplace(mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), i), field.type);
      return destructured;
    }
  }], "\n", extra2ClassDeclaration);
}

// A packed struct. Declares the P4HIR::StructType in C++.
def StructType : StructLikeType<"Struct", "struct"> {
  let summary = "struct type";
  let description = [{
    Represents a structure of name, value pairs.
    !p4hir.struct<"name", fieldName1: Type1, fieldName2: Type2>
  }];
}

//===----------------------------------------------------------------------===//
// ValidType: type used to represent validity bit in headers. We explicitly
// want it to be distinct from bool type to ensure we can always identify it
// by the type.
//===----------------------------------------------------------------------===//

def ValidBitType : P4HIR_Type<"ValidBit", "validity.bit"> {
  let summary = "Valid bit type";
  let description = [{
    `p4hir.valid.bit` represents valid bit in headers.
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "validity_bit"; };
  }];
}

//===----------------------------------------------------------------------===//
// HeaderType
//===----------------------------------------------------------------------===//

// A header. Declares the P4HIR::HeaderType in C++.
def HeaderType : StructLikeType<"Header", "header"> {
  let summary = "header type";
  let description = [{
    Represents a structure of name, value pairs.
    !p4hir.header<"name", fieldName1: Type1, fieldName2: Type2>

    Special field named "__valid" of type !p4hir.validity.bit is used to
    represent validity bit
  }];

  // We skip default builders entirely to consistently add validity bit field on fly
  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name,
                     "llvm::ArrayRef<P4HIR::FieldInfo>":$fields)>
  ];

  // This adds more C++ stuff into parent extraClassDeclaration
  let extra2ClassDeclaration = [{
    static constexpr llvm::StringRef validityBit = "__valid";
  }];
}

//===----------------------------------------------------------------------===//
// HeaderUnionType
//===----------------------------------------------------------------------===//

// A header union. Declares the P4HIR::HeaderUnionType in C++.
def HeaderUnionType : StructLikeType<"HeaderUnion", "header_union"> {
  let summary = "header union type";
  let description = [{
    A header union represents an alternative containing at most one of several different headers.
    !p4hir.header_union<"name", field1: HeaderType1, field2: HeaderType2>
  }];
}

//===----------------------------------------------------------------------===//
// EnumType, ErrorType  & SerEnumType
//===----------------------------------------------------------------------===//
def EnumType : P4HIR_Type<"Enum", "enum", []> {
  let summary = "enum type";
  let description = [{
    Represents an enumeration of values
    !p4hir.enum<"name", Case1, Case2>
  }];

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins StringRefParameter<"enum name">:$name,
                        "mlir::ArrayAttr":$fields);

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field) { return indexOf(field).has_value(); }

    /// Returns the index of the requested field, or a nullopt if the field is
    /// not part of this enum.
    std::optional<size_t> indexOf(mlir::StringRef field);
  }];
}

def ErrorType : P4HIR_Type<"Error", "error", []> {
  let summary = "error type";
  let description = [{
    Represents an enumeration of error values, essentially an enum
    !p4hir.error<Case1, Case2>
  }];

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins "mlir::ArrayAttr":$fields);

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field) { return indexOf(field).has_value(); }

    /// Returns the index of the requested field, or a nullopt if the field is
    /// not part of this enum.
    std::optional<size_t> indexOf(mlir::StringRef field);

    llvm::StringRef getAlias() const { return "error"; };
  }];
}

def SerEnumType : P4HIR_Type<"SerEnum", "ser_enum", []> {
  let summary = "serializable enum type";
  let description = [{
    Represents an enumeration of values backed by some integer value
    !p4hir.ser_enum<"name", !p4hir.bit<32>, Case1 : 42, Case2 : 0>
  }];

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins StringRefParameter<"enum name">:$name,
                        "P4HIR::BitsType":$type, "mlir::DictionaryAttr":$fields);

  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name,
                        "P4HIR::BitsType":$type, "mlir::DictionaryAttr":$fields), [{
      return $_get(type.getContext(), name, type, fields);
    }]>,
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name,
                        "P4HIR::BitsType":$type, "llvm::ArrayRef<mlir::NamedAttribute>":$fields), [{
      return $_get(type.getContext(), name, type,
                   DictionaryAttr::get(type.getContext(), fields));
    }]>

  ];

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field) { return getFields().contains(field); }

    /// Returns the underlying value of the requested field. Must be BitsAttr.
    mlir::Attribute valueOf(mlir::StringRef field) { return getFields().get(field); }
  }];
}


//===----------------------------------------------------------------------===//
// SetType
//===----------------------------------------------------------------------===//
def SetType : P4HIR_Type<"Set", "set", []> {
  let summary = "set type";
  let description = [{
    Represents a set of values
    !p4hir.set<!p4hir.bit<32>>
  }];

  let parameters = (ins "mlir::Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
    TypeBuilder<(ins "mlir::Type":$elementType), [{
      return $_get($_ctxt, elementType);
    }]>
  ];

  let assemblyFormat = [{
    `<` $elementType `>`
  }];

  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//
def TypeVarType : P4HIR_Type<"TypeVar", "type_var", []> {
  let summary = "type variable type";
  let description = [{
    Represents a type variable type:
    !p4hir.type_var<"T">
  }];

  let parameters = (ins StringRefParameter<"type variable name">:$name);

  // TODO: Check that it is only used in declarations
  let assemblyFormat = [{
    `<` $name `>`
  }];
}

//===----------------------------------------------------------------------===//
// ExternType
//===----------------------------------------------------------------------===//
def ExternType : P4HIR_Type<"Extern", "extern", []> {
  let summary = "extern type";
  let description = [{
    Represents a P4 extern:
    !p4hir.extern<"X"<!i32i>>
  }];

  let parameters = (ins StringRefParameter<"extern name">:$name,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments);
  let assemblyFormat = [{
    `<` $name (`<` $type_arguments^ `>`)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def AnyP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, StringType,
                           StructType, HeaderType, HeaderUnionType, Builtin_Tuple,
                           EnumType, SerEnumType,
                           ValidBitType,
                           DontcareType, ErrorType, UnknownType,
                           SetType]> {}
def AnyIntP4Type : AnyTypeOf<[BitsType, InfIntType]> {}
def CallResultP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, VoidType]> {}
def LoadableP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType,
                                StructType, HeaderType, HeaderUnionType, Builtin_Tuple,
                                EnumType, SerEnumType, ErrorType,
                                ValidBitType]> {}
def AnyEnumType : AnyTypeOf<[EnumType, SerEnumType]>;
def StructLikeType : AnyTypeOf<[StructType, HeaderType, HeaderUnionType]>;
def P4ObjectType : AnyTypeOf<[ParserType, ExternType]>;

/// A ref type with the specified constraints on the nested type.
class SpecificRefType<Type type> : ConfinedType<ReferenceType,
  [SubstLeaves<"$_self", "mlir::cast<P4HIR::ReferenceType>($_self).getObjectType()",
               type.predicate>],
  "ref of " # type.summary, "P4HIR::ReferenceType"
> {
  Type objectType = type;
}

def StructRefType : SpecificRefType<StructType>;
def StructLikeRefType : SpecificRefType<StructLikeType>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
