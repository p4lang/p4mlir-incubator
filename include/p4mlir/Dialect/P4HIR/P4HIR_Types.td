#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"

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
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def AnyP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType,
                           DontcareType, ErrorType, UnknownType]> {}
def CallResultP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType, VoidType]> {}
def LoadableP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType]> {}


#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
