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
}

//===----------------------------------------------------------------------===//
// "Singleton" types
//===----------------------------------------------------------------------===//

def DontcareType : P4HIR_Type<"Dontcare", "dontcare"> {}
// FIXME: Add string for error here & declarations
def ErrorType : P4HIR_Type<"Error", "error"> {}
def UnknownType : P4HIR_Type<"Unknown", "unknown"> {}

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
// ParameterType
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
}

def ParamType : P4HIR_Type<"Param", "param"> {
  let parameters = (ins "mlir::Type":$paramType,
                    DefaultValuedParameter<"ParamDirection", "ParamDirection::None">:$dir);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$paramType,
                                        CArg<"ParamDirection", "ParamDirection::None">:$dir), [{
      return $_get(paramType.getContext(), paramType, dir);
    }]>,
    TypeBuilder<(ins "mlir::Type":$paramType,
                     CArg<"ParamDirection", "ParamDirection::None">:$dir), [{
      return $_get($_ctxt, paramType, dir);
    }]>
  ];


  // Normally parameter types do not exist on their own, they are just part of
  // FuncType / ActionType. The latter use custom parsers / printers for parameter
  // types. Still, this one is useful for dumps.
  let assemblyFormat = [{
    `<` $paramType ( `,` $dir^ )? `>`
  }];

  let skipDefaultBuilders = 1;

  // FIXME: check that out and inout parameters are references and evething else
  // is value typed.
  // FIXME: refine paramType type constraint
}

//===----------------------------------------------------------------------===//
// ParameterType
//===----------------------------------------------------------------------===//

def P4HIR_ActionType : P4HIR_Type<"Action", "action"> {
  let summary = "P4 action type";
  let description = [{
    The `!p4hir.action` is an action type. It is essentially a list of parameter
    types. There is no return type for actions.

    Example:

    ```mlir
    !p4hir.action<()>
    !p4hir.action<(in !p4hir.bit<32>, out !p4hir.int<42>)>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"ParamType">:$inputs);

  // Use a custom parser to handle the argument types in better way.
  let assemblyFormat = [{
    `<` custom<ActionType>($inputs) `>`
  }];

  let extraClassDeclaration = [{
    /// Returns the `i`th input operand type. Asserts if out of bounds.
    P4HIR::ParamType getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Returns a clone of this action type with the given argument
    /// and result types.
    ActionType clone(mlir::TypeRange inputs) const;
  }];
}

//===----------------------------------------------------------------------===//
// P4HIR type constraints.
//===----------------------------------------------------------------------===//

def AnyP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType,
                           DontcareType, ErrorType, UnknownType]> {}
def LoadableP4Type : AnyTypeOf<[BitsType, BooleanType, InfIntType]> {}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_TD
