#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
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
// FIXME: Add string for error here & declarations
def ErrorType : P4HIR_Type<"Error", "error"> {}
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
                        "mlir::Type":$optionalReturnType);

  let builders = [
    // Construct with an actual return type or explicit !p4hir.void
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$inputs, "mlir::Type":$returnType), [{
      return $_get(returnType.getContext(), inputs,
                       mlir::isa<P4HIR::VoidType>(returnType) ? nullptr
                                                            : returnType);
    }]>,

    // Construct without return type
    TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$inputs), [{
      return $_get($_ctxt, inputs, nullptr);
    }]>

  ];

  // Use a custom parser to handle the argument types in better way.
  let assemblyFormat = [{
    `<` custom<FuncType>($optionalReturnType, $inputs) `>`
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
// StructType
//===----------------------------------------------------------------------===//

// A packed struct. Declares the P4HIR::StructType in C++.
def StructType : P4HIR_Type<"Struct", "struct", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>,
  DeclareTypeInterfaceMethods<FieldIDTypeInterface>
]> {
  let summary = "struct type";
  let description = [{
    Represents a structure of name, value pairs.
    !p4hir.struct<"name", fieldName1: Type1, fieldName2: Type2>
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let parameters = (
    ins StringRefParameter<"struct name">:$name, ArrayRefParameter<
      "P4HIR::StructType::FieldInfo", "struct fields">:$elements
  );

  let extraClassDeclaration = [{
    using FieldInfo = P4HIR::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type>&);
    std::optional<unsigned> getFieldIndex(mlir::StringRef fieldName);
    std::optional<unsigned> getFieldIndex(mlir::StringAttr fieldName);
  }];
}

//===----------------------------------------------------------------------===//
// EnumType & SerEnumType
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
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def AnyP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, StructType,
                           EnumType, SerEnumType,
                           DontcareType, ErrorType, UnknownType]> {}
def AnyIntP4Type : AnyTypeOf<[BitsType, InfIntType]> {}
def CallResultP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, VoidType]> {}
def LoadableP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, StructType,
                                EnumType, SerEnumType]> {}
def AnyEnumType : AnyTypeOf<[EnumType, SerEnumType]>;

/// A ref type with the specified constraints on the nested type.
class SpecificRefType<Type type> : ConfinedType<ReferenceType,
  [SubstLeaves<"$_self", "mlir::cast<P4HIR::ReferenceType>($_self).getObjectType()",
               type.predicate>],
  "ref of " # type.summary, "P4HIR::ReferenceType"
> {
  Type objectType = type;
}

def StructRefType : SpecificRefType<StructType>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
