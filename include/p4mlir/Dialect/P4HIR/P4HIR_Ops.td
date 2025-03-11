#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"

//===----------------------------------------------------------------------===//
// Base P4HIR operation definition.
//===----------------------------------------------------------------------===//

class P4HIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<P4HIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// P4HIR operation definitions.
//===----------------------------------------------------------------------===//

def ConstOp : P4HIR_Op<"const",
    [ConstantLike, Pure, AllTypesMatch<["value", "res"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let summary = "Defines a P4 constant";
    let description = [{
        The `p4hir.const` operation turns a literal into an SSA value.
        The data is attached to the operation as an attribute. It
        represents a constant declaration in P4.

        Example:

        ```mlir
        %0 = p4hir.const #p4hir.int<-128> : !p4hir.int<8>
        ```
    }];

    // The constant operation takes a constant value (via typed attributes) and
    // optional name.

    let arguments = (ins TypedAttrInterface:$value, OptionalAttr<StrAttr>:$name);

    // The constant operation returns a single value of AnyP4Type.
    let results = (outs AnyP4Type:$res);

    let assemblyFormat = [{
        (` ` `[` $name^ `]`)? $value attr-dict
    }];

    // Allow building a ConstantOp with no name, etc.
    let builders = [
      OpBuilder<(ins "mlir::TypedAttr":$value), [{
        build($_builder, $_state, value.getType(), value, {});
      }]>,
      OpBuilder<(ins "mlir::TypedAttr":$value, "const llvm::Twine &":$name), [{
        build($_builder, $_state,
              value.getType(), value, $_builder.getStringAttr(name));
      }]>

    ];

    let hasVerifier = 1;
}

def VariableOp : P4HIR_Op<"variable", [
                 DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
/*,
                 DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
                 DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>*/]> {
  let summary = "Defines a scope-local variable";
  let description = [{
    The `p4hir.alloca` operation defines a scope-local variable.

    The presence `init` attribute indicates that the local variable represented
    by this alloca was originally initialized in P4 source code. In such
    cases, the first use contains the initialization (e.g. via p4hir.store).

    The result type is a reference to the input's type.

    Example:

    ```mlir
    // bit<32> count = 3;
    %0 = p4hir.variable ["count", init] : <!p4hir.bit<32>>
    ...
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    UnitAttr:$init
  );

  let results = (outs Res<ReferenceType, ""/*,
                      [MemAlloc<AutomaticAllocationScopeResource>]*/>:$ref);

  let assemblyFormat = [{
    ` ` `[` $name
       (`,` `init` $init^)?
    `]`
     attr-dict `:` type($ref)
  }];

  let hasVerifier = 0;
}

def ReadOp : P4HIR_Op<"read", [
  TypesMatchWith<"type of 'result' matches object type of 'ref'",
                 "ref", "result",
                 "mlir::cast<P4HIR::ReferenceType>($_self).getObjectType()">,/*,
  DeclareOpInterfaceMethods<PromotableMemOpInterface>*/
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

  let summary = "Read value from variable";
  let description = [{
    `p4hir.read` reads a value given a variable reference
    backed up by a `p4hir.ref` type.

    Example:

    ```mlir

    // Read from local variable, reference in %0.
    %1 = p4hir.read %0 : <!p4hir.bit<32>>

    ```
  }];

  let arguments = (ins Arg<ReferenceType, "the reference to load from"/*,
                           [MemRead]*/>:$ref);
  // FIXME: Constraint result type
  let results = (outs LoadableP4Type:$result);

  let assemblyFormat = [{
    $ref `:` type($ref) attr-dict
  }];

  // FIXME: add verifier.
}

def AssignOp : P4HIR_Op<"assign", [
  TypesMatchWith<"type of 'value' matches object type of 'addr'",
                 "ref", "value",
                 "mlir::cast<ReferenceType>($_self).getObjectType()">,/*,
                 DeclareOpInterfaceMethods<PromotableMemOpInterface>*/]> {

  let summary = "Assign value to variable";
  let description = [{
    `p4hir.assign` stores a value (first operand) to the object referenced
    in the second operand.

    Example:

    ```mlir
    // Store a function argument to local variable, address in %0.
    p4hir.assign %arg0, %0 : <!p4hir.bit<32>>
    ```
  }];

  let arguments = (ins LoadableP4Type:$value,
                       Arg<ReferenceType, "the object to store the value"/*,
                           [MemWrite]*/>:$ref);

  let assemblyFormat = [{
    $value `,` $ref attr-dict `:` type($ref)
  }];

  // FIXME: add verifier.
}

// TODO: Decide if we'd want to be more precise and split cast into
// bitcast, trunc and extensions
// TODO: Add CastOpInterface
def CastOp : P4HIR_Op<"cast",
             [Pure/*,
              DeclareOpInterfaceMethods<PromotableOpInterface>*/,
              DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
              ]> {
  let summary = "Conversion between values of different types";
  let description = [{
    Apply P4 usual conversions rules between values.
    ```mlir
    %4 = p4hir.cast (%3 : !p4hir:bit<32>) : !p4hir:int<32>
    ```
  }];

  let arguments = (ins AnyP4Type:$src);
  let results = (outs AnyP4Type:$result);

  let assemblyFormat = [{
    `(` $src `:` type($src) `)`
    `:` type($result) attr-dict
  }];

  // FIXME: add verifier.
}

def UnaryOpKind_Neg   : I32EnumAttrCase<"Neg",   1, "minus">; // unary minus (-)
def UnaryOpKind_UPlus : I32EnumAttrCase<"UPlus", 2, "plus">;  // unary plus (+)
def UnaryOpKind_Cmpl  : I32EnumAttrCase<"Cmpl",  3, "cmpl">;  // complement (~)
def UnaryOpKind_LNot  : I32EnumAttrCase<"LNot",  4, "not">;   // logical not

def UnaryOpKind : I32EnumAttr<"UnaryOpKind",
  "unary operation (arith and logic) kind",
  [UnaryOpKind_Neg,  UnaryOpKind_UPlus,
   UnaryOpKind_Cmpl, UnaryOpKind_LNot]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def UnaryOp : P4HIR_Op<"unary",
  [Pure, SameOperandsAndResultType,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Unary operations";
  let description = [{
    `p4hir.unary` performs the unary operation according to
    the specified opcode kind: [minus, plus, cmpl, not].

    It requires one input operand and has one result, both types
    should be the same.

    ```mlir
    %7 = p4hir.unary(minus, %1) : !p4hir.bit<32>
    %8 = p4hir.unary(not, %2) : !p4hir.bit<32>
    ```
  }];

  let results = (outs AnyP4Type:$result);
  // FIXME: constraint types for unary operations
  // FIXME: check type for lnot
  let arguments = (ins Arg<UnaryOpKind, "unary op kind">:$kind, Arg<AnyP4Type>:$input);

  let assemblyFormat = [{
      `(` $kind `,` $input `)` `:` type($input) attr-dict
  }];

  let hasVerifier = 1;
}

def BinOpKind_Mul    : I32EnumAttrCase<"Mul",   1, "mul">;
def BinOpKind_Div    : I32EnumAttrCase<"Div",   2, "div">;
def BinOpKind_Mod    : I32EnumAttrCase<"Mod",   3, "mod">;
def BinOpKind_Add    : I32EnumAttrCase<"Add",   4, "add">;
def BinOpKind_Sub    : I32EnumAttrCase<"Sub",   5, "sub">;
def BinOpKind_AddSat : I32EnumAttrCase<"AddSat",6, "sadd">;
def BinOpKind_SubSat : I32EnumAttrCase<"SubSat",7, "ssub">;
def BinOpKind_BOr    : I32EnumAttrCase<"Or",    8, "or">;
def BinOpKind_BXor   : I32EnumAttrCase<"Xor",   9, "xor">;
def BinOpKind_BAnd   : I32EnumAttrCase<"And",   10,"and">;

def BinOpKind : I32EnumAttr<"BinOpKind",
  "binary operation (arith and logic) kind",
  [BinOpKind_Mul, BinOpKind_Div, BinOpKind_Mod,
   BinOpKind_Add, BinOpKind_Sub,
   BinOpKind_AddSat, BinOpKind_SubSat,
   BinOpKind_BOr, BinOpKind_BXor, BinOpKind_BAnd]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def BinOp : P4HIR_Op<"binop", [Pure,
  SameTypeOperands, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

  let summary = "Binary operations (arith and logic)";
  let description = [{
    p4hir.binop performs the binary operation according to
    the specified opcode kind: [mul, div, mod, add, sub,
    sadd, ssub, and, xor, or].

    It requires two input operands and has one result, all types
    should be the same.

    ```mlir
    %3 = p4hir.binop(add, %1, %2) : !p4hir.bit<32>
    %4 = p4hir.binop(mul, %1, %2) : !p4hir.bit<32>
    ```
  }];

  let results = (outs AnyIntP4Type:$result);
  let arguments = (ins Arg<BinOpKind, "binop kind">:$kind,
                       AnyIntP4Type:$lhs, AnyIntP4Type:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) attr-dict
  }];
}

def ConcatOp : P4HIR_Op<"concat", [Pure]> {

  let summary = "Concatenation of bit-strings and/or fixed-width signed integers";
  let description = [{
    `p4hir.concat` performs concatenation of bit-strings and/or fixed-width signed integers.

    The two operands must be either `p4hir.bit<N>` or `p4hir.int<N>`, and they can be of different signedness and width.
    The result has the same signedness as the left operand and the width equal to the sum of the two operands' width.
    In concatenation, the left operand is placed as the most significant bits.

    ```mlir
    %0 = p4hir.const #p4hir.int<3> : !p4hir.bit<5>
    %1 = p4hir.const #p4hir.int<4> : !p4hir.int<10>
    %2 = p4hir.concat(%0 : !p4hir.bit<5>, %0 : !p4hir.bit<5>) : !p4hir.bit<10>
    %3 = p4hir.concat(%1 : !p4hir.int<10>, %1 : !p4hir.int<10>) : !p4hir.int<20>
    %4 = p4hir.concat(%0: !p4hir.bit<5>, %1 : !p4hir.int<10>) : !p4hir.bit<15>
    %5 = p4hir.concat(%1 : !p4hir.int<5>, %0 : !p4hir.bit<10>) : !p4hir.int<15>
    ```
  }];

  let results = (outs BitsType:$result);
  let arguments = (ins BitsType:$lhs, BitsType:$rhs);

  // Custom builder to infer the result type with the proper width
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      auto lhsBits = lhs.getType().cast<BitsType>();
      auto rhsBits = rhs.getType().cast<BitsType>();
      auto resultWidth = lhsBits.getWidth() + rhsBits.getWidth();
      auto resultType = BitsType::get($_builder.getContext(), resultWidth, lhsBits.isSigned());
      build($_builder, $_state, resultType, lhs, rhs);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` `:` type($result) attr-dict
  }];
}

def ShlOp : P4HIR_Op<"shl",
    [Pure,
     TypesMatchWith<
       "the resulting type of an arithmetic shift operation must equal the type of the left-hand side operand",
       "result", "lhs", "$_self">]> {
  let summary = "Arithmetic shift left operation";
  let description = [{
    `p4hir.shl` performs arithmetic shift left.
    The left operand is the value to shift, the right operand is the shift amount.
    The result has the same type as the left operand.

    ```mlir
    %0 = p4hir.const #p4hir.int<1> : !p4hir.bit<32>
    %1 = p4hir.const #p4hir.int<2> : !p4hir.bit<8>
    %2 = p4hir.const #p4hir.int<3> : !p4hir.infint
    %3 = p4hir.shl(%0, %1 : !p4hir.bit<8>) : !p4hir.bit<32>
    %4 = p4hir.shl(%0, %2 : !p4hir.infint) : !p4hir.bit<32>
    ```
  }];

  let results = (outs AnyIntP4Type:$result);
  let arguments = (ins AnyIntP4Type:$lhs, AnyIntP4Type:$rhs);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `:` type($rhs) `)` `:` type($result) attr-dict
  }];
}

def ShrOp : P4HIR_Op<"shr",
    [Pure,
     TypesMatchWith<
       "the resulting type of an arithmetic shift operation must equal the type of the left-hand side operand",
       "result", "lhs", "$_self">]> {
  let summary = "Arithmetic shift right operation";
  let description = [{
    `p4hir.shr` performs arithmetic shift right.
    The left operand is the value to shift, the right operand is the shift amount.
    The result has the same type as the left operand.

    ```mlir
    %0 = p4hir.const #p4hir.int<1> : !p4hir.bit<32>
    %1 = p4hir.const #p4hir.int<2> : !p4hir.bit<8>
    %2 = p4hir.const #p4hir.int<3> : !p4hir.infint
    %3 = p4hir.shr(%0, %1 : !p4hir.bit<8>) : !p4hir.bit<32>
    %4 = p4hir.shr(%0, %2 : !p4hir.infint) : !p4hir.bit<32>
    ```
  }];

  let results = (outs AnyIntP4Type:$result);
  let arguments = (ins AnyIntP4Type:$lhs, AnyIntP4Type:$rhs);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `:` type($rhs) `)` `:` type($result) attr-dict
  }];
}

def CmpOpKind_LT : I32EnumAttrCase<"Lt", 1, "lt">;
def CmpOpKind_LE : I32EnumAttrCase<"Le", 2, "le">;
def CmpOpKind_GT : I32EnumAttrCase<"Gt", 3, "gt">;
def CmpOpKind_GE : I32EnumAttrCase<"Ge", 4, "ge">;
def CmpOpKind_EQ : I32EnumAttrCase<"Eq", 5, "eq">;
def CmpOpKind_NE : I32EnumAttrCase<"Ne", 6, "ne">;

def CmpOpKind : I32EnumAttr<
    "CmpOpKind",
    "compare operation kind",
    [CmpOpKind_LT, CmpOpKind_LE, CmpOpKind_GT,
     CmpOpKind_GE, CmpOpKind_EQ, CmpOpKind_NE]> {
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def CmpOp : P4HIR_Op<"cmp",
  [Pure, SameTypeOperands,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Compare values two values and produce a boolean result";
  let description = [{
    `p4hir.cmp` compares two input operands of the same type and produces a
    `p4hir.bool` result. The kinds of comparison available are:
    [lt,gt,ge,eq,ne]

    ```mlir
    %7 = p4hir.cmp(gt, %1, %2) : !p4hir.bit<32>, !p4hir.bool
    ```
  }];

  let results = (outs BooleanType:$result);
  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind,
                       AnyP4Type:$lhs, AnyP4Type:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

def ScopeOp : P4HIR_Op<"scope", [
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope,
       NoRegionArguments]> {
  let summary = "Represents a P4 scope";
  let description = [{
    `p4hir.scope` contains one region and defines a strict "scope" for all new
    values produced within its blocks.

    The region can contain an arbitrary number of blocks but usually defaults
    to one and can optionally return a value (useful for representing values
    coming out of expressions that are not parts of other expressions, short-cut
    logical operations and simplifying inlining logic) via `p4hir.yield`:


    ```mlir
    %rvalue = p4hir.scope {
      ...
      p4hir.yield %value
    }
    ```

    The blocks can be terminated by `p4hir.yield` or `p4hir.return`.
    If `p4hir.scope` yields no value, the `p4hir.yield` can be left out, and
    will be inserted implicitly.
  }];

  let results = (outs Optional<AnyP4Type>:$results);
  let regions = (region AnyRegion:$scopeRegion);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    custom<OmittedTerminatorRegion>($scopeRegion) (`:` type($results)^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Determine whether the scope is empty, meaning it contains a single block
    /// terminated by a p4hir.yield.
    bool isEmpty() {
      auto &entry = getScopeRegion().front();
      return getScopeRegion().hasOneBlock() &&
        llvm::isa<YieldOp>(entry.front());
      }
    }];

  let builders = [
    // Scopes for yielding values.
    OpBuilder<(ins
              "llvm::function_ref<void(mlir::OpBuilder &, mlir::Type &, mlir::Location)>":$scopeBuilder)>,
    // Scopes without yielding values.
    OpBuilder<(ins "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$scopeBuilder)>
  ];
}

def YieldOp : P4HIR_Op<"yield", [ReturnLike, Terminator,
    ParentOneOf<["ScopeOp", "TernaryOp", "IfOp",
                 "ParserSelectCaseOp",
                 // "SwitchOp", "CaseOp",
                 // "ForInOp", "ForOp",
                 ]>]> {
  let summary = "Represents the default branching behaviour of a region";
  let description = [{
    The `p4hir.yield` operation terminates regions on different P4HIR operations,
    and it is used to represent the default branching behaviour of a region.
    Said branching behaviour is determined by the parent operation. For
    example, a yield in a `p4hir.if` region implies a branch to the exit block,
    and so on.

    In some cases, it might yield an SSA value and the semantics of how the
    values are yielded is defined by the parent operation. For example, a
    `p4hir.ternary` operation yields a value from one of its regions.

    As a general rule, `p4hir.yield` must be explicitly used whenever a region has
    more than one block and no terminator, or within `p4hir.switch` regions not
    `p4hir.return` terminated.

    Examples:
    ```mlir
    p4hir.if %4 {
      ...
      p4hir.yield
    }

    p4hir.switch (%5) [
      case (equal, 3) {
        ...
        p4hir.yield
      }, ...
    ]

    p4hir.scope {
      ...
      p4hir.yield
    }

    %x = p4hir.scope {
      ...
      p4hir.yield %val
    }

    %y = p4hir.ternary {
      ...
      p4hir.yield %val : !p4hir.bit<42>
    } :  !p4hir.bit<42>
    ```
  }];

  let arguments = (ins Variadic<AnyP4Type>:$args);
  let assemblyFormat = "($args^ `:` type($args))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];
}

def TernaryOp : P4HIR_Op<"ternary",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "The `cond ? a : b` C/C++ ternary operation";
  let description = [{
    The `p4hir.ternary` operation represents operation that is absent in P4 language, but otherwise
    is useful to represent varios language-level constructs. It is essentialyl a C/C++ ternary. First
    argument is a `p4hir.bool` condition to evaluate, followed by two regions to execute (true or false).
    This is different from `p4hir.if` since each region is one block sized and the `p4hir.yield` closing the
    block scope should have one argument.

    Example:

    ```mlir
    // x = a && b;

    %x = p4hir.ternary (%a, true_region {
      ...
      p4hir.yield %b : !p4hir.bool
    }, false_region {
      ...
      p4hir.yield %a : !p4hir.bool
    }) -> !p4hir.bool
    ```
  }];
  let arguments = (ins BooleanType:$cond);
  let regions = (region AnyRegion:$trueRegion,
                        AnyRegion:$falseRegion);
  let results = (outs Optional<AnyP4Type>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$cond,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$trueBuilder,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$falseBuilder)
      >
  ];

  // All constraints already verified elsewhere.
  let hasVerifier = 0;

  let assemblyFormat = [{
    `(` $cond `,`
      `true` $trueRegion `,`
      `false` $falseRegion
    `)` `:` functional-type(operands, results) attr-dict
  }];
}

def IfOp : P4HIR_Op<"if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "The if-then-else operation";
  let description = [{
    The `p4hir.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand is a `p4hir.bool`
    type.

    Examples:

    ```mlir
    p4hir.if %b  {
      ...
    } else {
      ...
    }

    p4hir.if %c  {
      ...
    }
    ```

    `p4hir.if` defines no values and the 'else' can be omitted. The if/else
    regions must be terminated. If the region has only one block, the terminator
    can be left out, and `p4hir.yield` terminator will be inserted implictly.
    Otherwise, the region must be explicitly terminated.
  }];
  let arguments = (ins BooleanType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$cond, "bool":$withElseRegion,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "P4HIR::buildTerminatedBody">:$thenBuilder,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "nullptr">:$elseBuilder)>
  ];
}

def ReturnOp : P4HIR_Op<"return", [ParentOneOf<["ScopeOp", "IfOp",
                                                "FuncOp",
                                             // "SwitchOp", "CaseOp",
                                             // "ForInOp", "ForOp",
                                             ]>,
  // Note that ReturnOp is not ReturnLike: currently there is no way to
  // represent early exits in MLIR "properly"
  // We might not be able to have it a Terminator at this level in order
  // to represent dead code. We might lower it to proper terminator later (!)
  // See https://discourse.llvm.org/t/rfc-region-based-control-flow-with-early-exits-in-mlir/76998
                                   Terminator]> {
  let summary = "Return from function or action";
  let description = [{
    The "return" operation represents a return operation within a function or action.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.

    ```mlir
      p4hir.action @foo() {
        ...
        p4hir.return
      }

      p4hir.func @foo() -> !p4hir.bit<32> {
        ...
        p4hir.return %0 : !p4hir.bit<32>
      }

    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyP4Type>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

def FuncOp : P4HIR_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove
]> {
  let summary = "Define a function-like object (action, function)";
  let description = [{
    Similar to `mlir::FuncOp` built-in:
    > Operations within the function cannot implicitly capture values defined
    > outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    > external references must use function arguments or attributes that establish
    > a symbolic connection (e.g. symbols referenced by name via a string
    > attribute like SymbolRefAttr). An external function declaration (used when
    > referring to a function declared in some other module) has no body. While
    > the MLIR textual form provides a nice inline syntax for function arguments,
    > they are internally represented as “block arguments” to the first block in
    > the region.
    >
    > Only dialect attribute names may be specified in the attribute dictionaries
    > for function arguments, results, or the function itself.

    Parameters might have direction that is specified via `p4hir.dir`
    attribute. Out and inout parameters must have a reference type. All
    refence-typed parameters must have a direction and it should be `out` or `input`.

    An action must be marked as `action`, should always have a body and cannot return
    anything.

    Example:

    ```mlir
    // Action with undirection parameter
    p4hir.action @foo(%arg3 : !p4hir.int<42>) {
      ...
    }

    // Action with direction parameters
    p4hir.action @foo(%arg0 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir inout>},
                  %arg1 : !bit32 {p4hir.dir = #p4hir<dir in>},
                  %arg2 : !p4hir.ref<!bit32> {p4hir.dir = #p4hir<dir out>},
                  %arg3 : !p4hir.int<42>) {
      ...
    }

    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FuncType>:$function_type,
                       UnitAttr:$action,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<ArrayAttr>:$annotations);
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "FuncType":$type,
               CArg<"bool", "false">:$isExternal,
               CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs,
               CArg<"llvm::ArrayRef<mlir::DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // TODO: move to custom builder
    static FuncOp buildAction(mlir::OpBuilder &builder,
                              mlir::Location loc,
                              llvm::StringRef name,
                              P4HIR::FuncType type,
                              llvm::ArrayRef<mlir::NamedAttribute> attrs = {},
                              llvm::ArrayRef<mlir::DictionaryAttr> argAttrs = {}) {
      auto op = builder.create<FuncOp>(loc, name, type, false, attrs, argAttrs);
      op.createEntryBlock();
      op.setAction(true);
      return op;
    }

    /// Returns the region on the current operation that is callable. Always
    /// non-null for actions.
    mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getFunctionType().getReturnTypes();
    }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or null if
    /// there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    /// Returns the argument types of this action.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
       return getFunctionType().getInputs();
    }

    /// Returns the `i`th argument type. Asserts if out of bounds.
    mlir::Type getArgumentType(unsigned i) {
      return getArgumentTypes()[i];
    }

    /// Returns attribute name used to store directions
    static llvm::StringRef getDirectionAttrName() { return "p4hir.dir"; }

    /// Return the `i`th argument direction.
    ParamDirection getArgumentDirection(unsigned i) {
      mlir::ArrayAttr argAttrs = getArgAttrsAttr();
      if (!argAttrs)
        return ParamDirection::None;

      if (auto dirAttr =
            mlir::cast<mlir::DictionaryAttr>(argAttrs[i]).get(getDirectionAttrName()))
         return mlir::cast<ParamDirectionAttr>(dirAttr).getValue();

      return ParamDirection::None;
    }

    /// Returns 0 or 1 result type of this function (0 in the case of a function
    /// returing void or action)
    llvm::ArrayRef<mlir::Type> getResultTypes() {
       return getFunctionType().getReturnTypes();
    }

    /// Hook for OpTrait::FunctionOpInterfaceTrait, called after verifying that
    /// the 'type' attribute is present and checks if it holds a function type.
    /// Ensures getType, getNumFuncArguments, and getNumFuncResults can be
    /// called safely.
    llvm::LogicalResult verifyType();

    bool isDeclaration() { return isExternal(); }

    void createEntryBlock();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def CallOp : P4HIR_Op<"call",
  [NoRegionArguments, CallOpInterface,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a method (action, function, etc.) that
    is within the same symbol scope as the call. The operands and result types of the call must
    match the specified function type. The callee is encoded as a symbol reference attribute
    named "callee".

    Example:

    ```mlir
    // Direct call of function
    %2 = p4hir.call @my_add(%0, %1) : (!p4hir.bit<8>, !p4hir.bit<8>) -> !p4hir.bit<8>
    // Direct call of action
    %4 = p4hir.call @my_add(%0, %1) : (!p4hir.bit<8>, !p4hir.bit<8>) -> ()
    ...
    ```
  }];

  // TODO: Refine result types, refine parameter type
  let results = (outs Optional<CallResultP4Type>:$result);
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<TypeArrayAttr>:$type_operands,
                       Variadic<AnyType>:$arg_operands);

  let skipDefaultBuilders = 1;
  let hasVerifier = 0;

  let builders = [
    // Functions
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::Type":$resType,
               "mlir::TypeRange":$type_operands,
               "mlir::ValueRange":$arg_operands), [{
      $_state.addOperands(arg_operands);
      $_state.addAttribute("callee", callee);
      if (resType && !isa<VoidType>(resType))
        $_state.addTypes(resType);
      if (!type_operands.empty())
        $_state.addAttribute("type_operands", $_builder.getTypeArrayAttr(type_operands));
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::Type":$resType,
               "mlir::ValueRange":$arg_operands), [{
      $_state.addOperands(arg_operands);
      $_state.addAttribute("callee", callee);
      if (resType && !isa<VoidType>(resType))
        $_state.addTypes(resType);
    }]>,
    // Everything else that does not produce result
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee,
                   "mlir::TypeRange":$type_operands,
                   "mlir::ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      if (!type_operands.empty())
        $_state.addAttribute("type_operands", $_builder.getTypeArrayAttr(type_operands));
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee,
                   "mlir::ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }

    void setArg(unsigned index, mlir::Value value) {
      setOperand(index, value);
    }
  }];

  let assemblyFormat = [{
    $callee (`<` $type_operands^ `>`)?`(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)
  }];
}

def StructOp : P4HIR_Op<"struct",
  [Pure,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a struct from constituent parts.";
  // FIXME: Better constraint type
  let arguments = (ins Variadic<AnyP4Type>:$input);
  let results = (outs StructLikeType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def StructExtractOp : P4HIR_Op<"struct_extract",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Extract a named field from a struct.";
  let description = [{
    ```
    %result = p4hir.struct_extract %input["field"] : !p4hir.struct<field: type>
    TODO: Support the nested extractions
    %result2 = p4hir.struct_extract %input["field1", "field2"] : !p4hir.struct<field1: !p4hir.struct<field2: type>>

    ```
  }];

  let arguments = (ins StructLikeType:$input, I32Attr:$fieldIndex);
  // FIXME: Better constraint type
  let results = (outs AnyP4Type:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "P4HIR::FieldInfo":$field)>,
    OpBuilder<(ins "mlir::Value":$input, "mlir::StringAttr":$fieldName)>,
    OpBuilder<(ins "mlir::Value":$input, "llvm::StringRef":$fieldName), [{
      build($_builder, $_state, input, $_builder.getStringAttr(fieldName));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    mlir::StringAttr getFieldNameAttr() {
      auto type = mlir::cast<P4HIR::StructLikeTypeInterface>(getInput().getType());
      return type.getFields()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    llvm::StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

def StructExtractRefOp : P4HIR_Op<"struct_extract_ref",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Project from a struct reference to a reference to a named struct field";
  let description = [{
    ```
    %result = p4hir.struct_extract_ref %input["field"] : <!p4hir.struct<field: type>>
    TODO: Support the nested extractions
    %result2 = p4hir.struct_extract_ref %input["field1", "field2"] : <!p4hir.struct<field1: !p4hir.struct<field2: type>>>

    ```
  }];

  let arguments = (ins StructLikeRefType:$input, I32Attr:$fieldIndex);
  let results = (outs ReferenceType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "P4HIR::FieldInfo":$field)>,
    OpBuilder<(ins "mlir::Value":$input, "mlir::StringAttr":$fieldName)>,
    OpBuilder<(ins "mlir::Value":$input, "llvm::StringRef":$fieldName), [{
      build($_builder, $_state, input, $_builder.getStringAttr(fieldName));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    mlir::StringAttr getFieldNameAttr() {
      auto type = mlir::cast<StructLikeTypeInterface>(mlir::cast<ReferenceType>(getInput().getType()).getObjectType());
      return type.getFields()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    llvm::StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

def TupleOp : P4HIR_Op<"tuple",
  [Pure,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a tuple from constituent parts.";
  // FIXME: Better constraint type
  let arguments = (ins Variadic<AnyP4Type>:$input);
  let results = (outs Builtin_Tuple:$result);
  let hasCustomAssemblyFormat = 1;
  // FIXME: use declarative format
  // let assemblyFormat = [{
  //   `(` $input `)` attr-dict `:` type($result)
  // }];

  let hasVerifier = 1;
}

def TupleExtractOp : P4HIR_Op<"tuple_extract",
    [Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Extract a field from a tuple.";
  let description = [{
    ```
    %result = p4hir.tuple_extract %input[0] : !tuple<!bit32, !bit64>

    ```
  }];

  // TODO: find a way to use fieldid inteerface on tuples...
  let arguments = (ins Builtin_Tuple:$input, I32Attr:$fieldIndex);
  // FIXME: Better constraint type
  let results = (outs AnyP4Type:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "unsigned":$fieldIndex)>,
    OpBuilder<(ins "mlir::Value":$input, "P4HIR::IntAttr":$fieldIndex), [{
      build($_builder, $_state, input, fieldIndex.getUInt());
    }]>
  ];

  // FIXME: use declarative format
  // let assemblyFormat = [{
  //   $input `[` $fieldIndex `]` attr-dict `:` type($result)
  // }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def InstantiateOp : P4HIR_Op<"instantiate",
  [NoRegionArguments, CallOpInterface,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "constructor call operation";
  let description = [{
    Instantiate P4 object: extern, parser, control, extern or package.

    Example:

    ```mlir
    // Instantiate a subparser
    %sp = p4hir.instantiate @subparser() as "sp" : () -> !p4hir.parser<"subparser", (!empty)>
    // Instantiate a subparser with ctor parameters
    %false = p4hir.const #false
    %sp2 = p4hir.instantiate @subparser2(%false) as "sp2" : (!p4hir.bool) -> !p4hir.parser<"subparser2", (!empty)>
    ...
    ```
  }];

  // TODO: Refine result types, refine parameter type (must be constants!)
  let results = (outs P4ObjectType:$result);
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$arg_operands,
                       OptionalAttr<StrAttr>:$name);

  // TODO: Add verifier
  let hasVerifier = 0;

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }

    void setArg(unsigned index, mlir::Value value) {
      setOperand(index, value);
    }
  }];

  let assemblyFormat = [{
    $callee `(` $arg_operands `)` `as` $name attr-dict `:` functional-type($arg_operands, results)
  }];
}

def ApplyOp : P4HIR_Op<"apply",
  [NoRegionArguments, CallOpInterface,
   //DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    TypesMatchWith<
       "the apply type of the object matches the types of operands",
       "callee", "arg_operands", "cast<P4HIR::ParserType>($_self).getInputs()">,
   ]> {
  let summary = "apply operation";
  let description = [{
    Applies P4 parser, table or control

    Example:

    ```mlir
    // Application of a subparser
    %sp = p4hir.instantiate @subparser() as "sp" : () -> !p4hir.parser<"subparser", ()>
    p4hir.apply %sp() : !p4hir.parser<"subparser", ()>
    ...
    ```
  }];

  // TODO: Refine result types, refine parameter type (must be constants!)
  let results = (outs);
  let arguments = (ins P4ObjectType:$callee, Variadic<AnyType>:$arg_operands);

  // TODO: Add verifier
  let hasVerifier = 0;

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getCallee();
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      getCalleeMutable().assign(callee.get<mlir::Value>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $arg_operands `)` attr-dict `:` type($callee)
  }];
}

def ExternOp : P4HIR_Op<"extern",
  [Symbol, SymbolTable, NoTerminator]> {
  let summary = "P4 extern declaration";
  let description = [{
    Declares P4 extern

    Example:

    ```mlir
      p4hir.extern @packet_in {
        p4hir.overload_set @extract {
          p4hir.func @extract_0(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out})
          p4hir.func @extract_1(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out}, !b32i {p4hir.dir = #in})
        }
        p4hir.func @lookahead() -> !p4hir.type_var<"T">
        p4hir.func @advance(!b32i {p4hir.dir = #in})
        p4hir.func @length() -> !b32i
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<ArrayAttr>:$type_parameters);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name (`<` $type_parameters^ `>`)? $body attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::Block &createEntryBlock();
  }];
}

def CallMethodOp : P4HIR_Op<"call_method",
  [NoRegionArguments, CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call method of an P4 exern";
  let description = [{
    Calls named method of a given P4 extern.

    Example:

    ```mlir
      p4hir.parser @parserI(%arg0: !p4hir.extern<"packet_in">, %arg1: !p4hir.ref<!Parsed_packet>)() {
        p4hir.state @start {
          %ethernet_field_ref = p4hir.struct_extract_ref %arg1["ethernet"] : <!Parsed_packet>
          %hdr_out_arg = p4hir.variable ["hdr_out_arg"] : <!Ethernet_h>
          p4hir.call_method @packet_in::@extract<[!Ethernet_h]> (%arg0, %hdr_out_arg) : !p4hir.extern<"packet_in">, (!p4hir.ref<!Ethernet_h>) -> ()
          ...
    ```
  }];

  // TODO: Refine result types, refine parameter type
  let results = (outs Optional<CallResultP4Type>:$result);
  let arguments = (ins ExternType:$base, SymbolRefAttr:$callee,
                       OptionalAttr<TypeArrayAttr>:$type_operands,
                       Variadic<AnyType>:$arg_operands);

  // let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
      $callee  (`<` $type_operands^ `>`)? `(`$base `,` $arg_operands `)` `:` qualified(type($base)) `,` functional-type($arg_operands, results) attr-dict
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$base, "mlir::SymbolRefAttr":$callee,
                   "mlir::TypeRange":$type_operands, "mlir::ValueRange":$arg_operands), [{
      $_state.addOperands(base);
      $_state.addOperands(arg_operands);
      $_state.addAttribute("callee", callee);
      if (resultType && !isa<VoidType>(resultType))
        $_state.addTypes(resultType);
      if (!type_operands.empty())
        $_state.addAttribute("type_operands", $_builder.getTypeArrayAttr(type_operands));
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getBase();
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      getBaseMutable().assign(callee.get<mlir::Value>());
    }
  }];
}

def OverloadSetOp : P4HIR_Op<"overload_set",
  [Symbol, SymbolTable, NoTerminator]> {
  let summary = "P4 extern method overload set";
  let description = [{
    Declares P4 a set of overloaded extern methods

    Example:
    ```mlir
      p4hir.extern @packet_in {
        p4hir.overload_set @extract {
          p4hir.func @extract_0(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out})
          p4hir.func @extract_1(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out}, !b32i {p4hir.dir = #in})
        }
        p4hir.func @lookahead() -> !p4hir.type_var<"T">
        p4hir.func @advance(!b32i {p4hir.dir = #in})
        p4hir.func @length() -> !b32i
      }
    ```
  }];

  // TODO: Implement overload set members checking
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::Block &createEntryBlock();
  }];
}


include "p4mlir/Dialect/P4HIR/P4HIR_ParserOps.td"

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
