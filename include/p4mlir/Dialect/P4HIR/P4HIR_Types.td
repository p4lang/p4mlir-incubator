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

def BitsType : P4HIR_TypeNoMnemonic<"Bits", "bits", [HasDefaultValue]> {
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

    mlir::TypedAttr getDefaultValue();
  }];
}

// Unsigned integer type of a specific width (bits<width>).
class Bit<int width>
  : Type<And<[
        CPred<"::mlir::isa<P4HIR::BitsType>($_self)">,
        CPred<"::mlir::cast<P4HIR::BitsType>($_self).isUnsigned()">,
        CPred<"::mlir::cast<P4HIR::BitsType>($_self).getWidth() == " # width>
        ]>, width # "-bit unsigned integer", "P4HIR::BitsType">,
    BuildableType<
      "P4HIR::BitsType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

// Signed integer type of a specific width (int<width>).
class Int<int width>
  : Type<And<[
        CPred<"::mlir::isa<P4HIR::IntType>($_self)">,
        CPred<"::mlir::cast<P4HIR::IntType>($_self).isSigned()">,
        CPred<"::mlir::cast<P4HIR::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit signed integer", "P4HIR::BitsType">,
    BuildableType<
      "p4hir::BitsType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

def InfIntType : P4HIR_Type<"InfInt", "infint", [HasDefaultValue]> {
  let summary = "arbitrary precision integer type";
  let description = [{
    `p4hir.infint` represents a P4 `int` type.
  }];

  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "infint"; };

    mlir::TypedAttr getDefaultValue();
  }];
}

//===----------------------------------------------------------------------===//
// VarBitsType: Bit-strings of dynamically-computed width
// with a fixed maximum width
//===----------------------------------------------------------------------===//

def VarBitsType : P4HIR_Type<"VarBits", "varbit", [HasDefaultValue]> {
  let summary = "bit-strings of dynamically-computed width with a fixed maximum width";
  let description = [{
    A variable-width bit-string type with a fixed maximum width.
    The actual width is determined at runtime but cannot exceed the maximum.
  }];

  let parameters = (ins "unsigned":$maxWidth);
  let assemblyFormat = "`<` $maxWidth `>`";

  let extraClassDeclaration = [{
    mlir::TypedAttr getDefaultValue();
  }];
}

//===----------------------------------------------------------------------===//
// BooleanType: An alternative here is to represent bool as mlir::i1, but
// let's be more generic, it could be lowered later on
//===----------------------------------------------------------------------===//

def BooleanType : P4HIR_Type<"Bool", "bool", [HasDefaultValue]> {
  let summary = "boolean type";
  let description = [{
    `p4hir.bool` represents a P4 `bool` type.
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "bool"; };
    mlir::TypedAttr getDefaultValue();
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
def ParserType : P4HIR_Type<"Parser", "parser", [AnnotatedType]> {
  let parameters = (ins StringRefParameter<"parser type name">:$name,
                        ArrayRefParameter<"mlir::Type">:$inputs,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$inputs,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
       return get($_ctxt, name, inputs, llvm::ArrayRef<mlir::Type>(), annotations);
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$inputs,
                     "llvm::ArrayRef<mlir::Type>":$type_arguments,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, inputs, type_arguments,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let assemblyFormat = [{
    `<` $name (`<` $type_arguments^ `>`)? (`annotations` $annotations^)?  `,` custom<FuncType>($inputs) `>`
  }];
}

//===----------------------------------------------------------------------===//
// ControlType
//===----------------------------------------------------------------------===//
def ControlType : P4HIR_Type<"Control", "control", [AnnotatedType]> {
  let parameters = (ins StringRefParameter<"control type name">:$name,
                        ArrayRefParameter<"mlir::Type">:$inputs,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$inputs,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
       return get($_ctxt, name, inputs, llvm::ArrayRef<mlir::Type>(), annotations);
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$inputs,
                     "llvm::ArrayRef<mlir::Type>":$type_arguments,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, inputs, type_arguments,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let assemblyFormat = [{
    `<` $name (`<` $type_arguments^ `>`)? (`annotations` $annotations^)? `,` custom<FuncType>($inputs) `>`
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
  AnnotatedType,
  HasDefaultValue
]> {
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let parameters = (
    ins StringRefParameter<"struct name">:$name,
        ArrayRefParameter<"P4HIR::FieldInfo", "struct fields">:$elements,
        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations
  );

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name,
                     "llvm::ArrayRef<P4HIR::FieldInfo>":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, fields,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  string commonClassDeclaration = [{
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

    llvm::ArrayRef<P4HIR::FieldInfo> getFields() {
      return getElements();
    }

    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type> &types) {
      for (const auto &field : getElements()) types.push_back(field.type);
    }

    P4HIR::FieldInfo getFieldByID(unsigned fieldID) {
      assert((fieldID != 0) && "invalid fieldID");
      mlir::Type type = *this;
      const P4HIR::FieldInfo *nestedField = nullptr;
      while (fieldID) {
        auto stype = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(type);
        if (stype) {
          auto [index, newFieldID] = stype.getIndexAndSubfieldID(fieldID);
          nestedField = &stype.getFields()[index];
          type = nestedField->type;
          fieldID = newFieldID;
        } else {
          llvm::report_fatal_error("fieldID indexing into a non-struct type");
        }
      }
      return *nestedField;
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

    // DestructurableTypeInterface implementation
    mlir::Type getTypeAtIndex(mlir::Attribute index) const {
      auto indexAttr = mlir::dyn_cast<mlir::IntegerAttr>(index);
      if (!indexAttr) return {};

      return getElements()[indexAttr.getInt()].type;
    }

    std::optional<llvm::DenseMap<mlir::Attribute, mlir::Type>> getSubelementIndexMap() const {
      llvm::DenseMap<mlir::Attribute, mlir::Type> destructured;
      for (auto [i, field] : llvm::enumerate(getElements()))
          destructured.try_emplace(mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), i), field.type);
      return destructured;
    }

    // HasDefaultValue interface implementation
    mlir::TypedAttr getDefaultValue() { return getStructLikeDefaultValue(*this); }
  }];
  let extraClassDeclaration = commonClassDeclaration;
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

def ValidBitType : P4HIR_Type<"ValidBit", "validity.bit", [HasDefaultValue]> {
  let summary = "Valid bit type";
  let description = [{
    `p4hir.valid.bit` represents valid bit in headers.
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    llvm::StringRef getAlias() const { return "validity_bit"; };
    mlir::TypedAttr getDefaultValue();
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
                     "llvm::ArrayRef<P4HIR::FieldInfo>":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations)>
  ];

  // This adds more C++ stuff into parent extraClassDeclaration
  let extraClassDeclaration = commonClassDeclaration # [{
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
// HeaderStackType
//===----------------------------------------------------------------------===//
def HeaderStackType : StructLikeType<"HeaderStack", "header_stack"> {
  let summary = "header stack type";
  let description = [{
    A header stack is a fixed-size array of headers with the same type.
    The valid elements of a header stack need not be contiguous.
    !p4hir.header_stack<n x hdr, __nextIndex : !p4hir.bit<32>>

    We model header stack as an struct containing "data" field of array
    type and a field "__nextIndex" of type !p4hir.bit<32> that is used
    to represent next valid index.
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilder<(ins "size_t":$size, "P4HIR::StructLikeTypeInterface":$elementType)>
  ];

  let extraClassDeclaration = commonClassDeclaration # [{
    static constexpr llvm::StringRef dataFieldName = "data";
    static constexpr llvm::StringRef nextIndexFieldName = "nextIndex";

    ArrayType getDataType();
    size_t getArraySize();
    StructLikeTypeInterface getArrayElementType();
  }];
}


//===----------------------------------------------------------------------===//
// EnumType, ErrorType  & SerEnumType
//===----------------------------------------------------------------------===//
def EnumType : P4HIR_Type<"Enum", "enum", [AnnotatedType, HasDefaultValue]> {
  let summary = "enum type";
  let description = [{
    Represents an enumeration of values
    !p4hir.enum<"name", Case1, Case2>
  }];

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins StringRefParameter<"enum name">:$name,
                        "mlir::ArrayAttr":$fields,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
     TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Attribute>":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, mlir::ArrayAttr::get($_ctxt, fields),
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,

     TypeBuilder<(ins "llvm::ArrayRef<mlir::Attribute>":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, "", mlir::ArrayAttr::get($_ctxt, fields),
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,

    TypeBuilder<(ins "llvm::StringRef":$name, "mlir::ArrayAttr":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, fields,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,

    TypeBuilder<(ins "mlir::ArrayAttr":$fields,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, "", fields,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field) { return indexOf(field).has_value(); }

    /// Returns the index of the requested field, or a nullopt if the field is
    /// not part of this enum.
    std::optional<size_t> indexOf(mlir::StringRef field);

    mlir::TypedAttr getDefaultValue();
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

def SerEnumType : P4HIR_Type<"SerEnum", "ser_enum", [AnnotatedType, HasDefaultValue]> {
  let summary = "serializable enum type";
  let description = [{
    Represents an enumeration of values backed by some integer value
    !p4hir.ser_enum<"name", !p4hir.bit<32>, Case1 : 42, Case2 : 0>
  }];

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins StringRefParameter<"enum name">:$name,
                        "P4HIR::BitsType":$type, "mlir::DictionaryAttr":$fields,
                         OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name,
                        "P4HIR::BitsType":$type, "mlir::DictionaryAttr":$fields,
                        CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get(type.getContext(), name, type, fields,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name,
                        "P4HIR::BitsType":$type, "llvm::ArrayRef<mlir::NamedAttribute>":$fields,
                        CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get(type.getContext(), name, type,
                   DictionaryAttr::get(type.getContext(), fields),
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field) { return getFields().contains(field); }

    /// Returns the underlying value of the requested field. Must be IntAttr.
    mlir::Attribute valueOf(mlir::StringRef field) { return getFields().get(field); }
    mlir::Attribute valueOf(mlir::StringAttr field) { return getFields().get(field); }

    template<class T>
    T valueOf(mlir::StringRef field) { return getFields().getAs<T>(field); }
    template<class T>
    T valueOf(mlir::StringAttr field) { return getFields().getAs<T>(field); }

    mlir::TypedAttr getDefaultValue();
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
def ExternType : P4HIR_Type<"Extern", "extern", [AnnotatedType]> {
  let summary = "extern type";
  let description = [{
    Represents a P4 extern:
    !p4hir.extern<"X"<!i32i>>
  }];

  let parameters = (ins StringRefParameter<"extern name">:$name,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name, CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
       return get($_ctxt, name, llvm::ArrayRef<mlir::Type>(), annotations);
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$type_arguments,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, type_arguments,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let assemblyFormat = [{
    `<` $name (`<` $type_arguments^ `>`)? (`annotations` $annotations^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// PackageType
//===----------------------------------------------------------------------===//
def PackageType : P4HIR_Type<"Package", "package", [AnnotatedType]> {
  let summary = "package type";
  let description = [{
    Represents a P4 package:
    !p4hir.package<"X"<!i32i>>
  }];

  let parameters = (ins StringRefParameter<"package name">:$name,
                        OptionalArrayRefParameter<"mlir::Type">:$type_arguments,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name, "llvm::ArrayRef<mlir::Type>":$type_arguments,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, type_arguments,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let assemblyFormat = [{
    `<` $name (`<` $type_arguments^ `>`)? (`annotations` $annotations^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// AliasType
//===----------------------------------------------------------------------===//
def AliasType : P4HIR_Type<"Alias", "alias", [AnnotatedType]> {
  let summary = "alias type";
  let description = [{
    Represents a P4 "new type" that is an alias to the existing one:
    !p4hir.alias<!i32i>
  }];

  let parameters = (ins StringRefParameter<"type name">:$name, "mlir::Type":$aliased_type,
                        OptionalParameter<"mlir::DictionaryAttr", "annotations">:$annotations);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name, "mlir::Type":$aliased_type,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get(aliased_type.getContext(), name, aliased_type,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$aliased_type,
                     CArg<"mlir::DictionaryAttr", "{}">:$annotations), [{
      return $_get($_ctxt, name, aliased_type,
                   annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
    }]>
  ];

  let assemblyFormat = [{
    `<` $name  (`annotations` $annotations^)? `,` $aliased_type `>`
  }];

  let extraClassDeclaration = [{
    /// Gets the fully resolved type by following the alias chain.
    mlir::Type getCanonicalType() {
      mlir::Type type = getAliasedType();
      while (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type)) {
        type = aliasType.getAliasedType();
      }
      return type;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//
def ArrayType : P4HIR_Type<"Array", "array", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>,
  DeclareTypeInterfaceMethods<FieldIDTypeInterface>,
  HasDefaultValue
]> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized arrays are roughly similar to C arrays. Arrays are always considered
    packed.
  }];

  let mnemonic = "array";
  let parameters = (ins "size_t":$size, "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<
      (ins "size_t":$size, "mlir::Type":$elementType), [{
        return $_get(elementType.getContext(), size, elementType);
      }]>
  ];

  let assemblyFormat = [{
    `<` custom<Array>($size, $elementType) `>`
  }];

  let extraClassDeclaration = [{
    mlir::TypedAttr getDefaultValue();
  }];
}

//===----------------------------------------------------------------------===//
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def AnyP4Type : AnyTypeOf<[BitsType, VarBitsType, BooleanType, InfIntType, StringType,
                           StructType, HeaderType, HeaderUnionType, HeaderStackType, Builtin_Tuple,
                           EnumType, SerEnumType,
                           ValidBitType,
                           DontcareType, ErrorType, UnknownType, AliasType,
                           SetType, ArrayType]> {}
def AnyIntP4Type : AnyTypeOf<[BitsType, InfIntType]> {}
def CallResultP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, VoidType,
                                  StructType, HeaderType, Builtin_Tuple,
                                  EnumType, SerEnumType, AliasType,
                                  ExternType]> {}
def LoadableP4Type : AnyTypeOf<[BitsType, VarBitsType, BooleanType, InfIntType,
                                StructType, HeaderType, HeaderUnionType, HeaderStackType, Builtin_Tuple,
                                EnumType, SerEnumType, ErrorType,
                                ValidBitType, AliasType, ArrayType]> {}
def AnyEnumType : AnyTypeOf<[EnumType, SerEnumType]>;
def AnyCollectionType : AnyTypeOf<[HeaderStackType, SetType, ArrayType]> {}
def SwitchCondType : AnyTypeOf<[AnyIntP4Type, AnyEnumType, ErrorType]>;
def StructLikeType : AnyTypeOf<[StructType, HeaderType, HeaderUnionType, HeaderStackType]>;
def P4ObjectType : AnyTypeOf<[ParserType, ControlType, ExternType, PackageType]>;

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
def BitsRefType : SpecificRefType<BitsType>;
def HeaderRefType : SpecificRefType<HeaderType>;
def ArrayRefType : SpecificRefType<ArrayType>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
