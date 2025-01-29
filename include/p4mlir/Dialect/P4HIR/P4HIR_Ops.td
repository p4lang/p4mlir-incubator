#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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
    [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]> {
    let summary = "Defines a P4 constant";
    let description = [{
        The `p4hir.const` operation turns a literal into an SSA value.
        The data is attached to the operation as an attribute. It
        represents a constant declaration in P4.

        Example:

        ```mlir
        %0 = p4hir.const p4hir.int<-128> : !p4hir.int<8>
        ```
    }];

    // The constant operation takes an attribute as the only input.
    let arguments = (ins TypedAttrInterface:$value);

    // The constant operation returns a single value of AnyP4Type.
    let results = (outs AnyP4Type:$res);

    let assemblyFormat = "attr-dict $value";

    let hasVerifier = 1;
}

class AllocaTypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : PredOpTrait<summary, CPred<
      comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
             ", $" # rhsArg # ")">> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

def AllocaOp : P4HIR_Op<"alloca", [
  AllocaTypesMatchWith<"'objectType' matches referenced type of 'object'",
                 "ref", "objectType",
                 "mlir::cast<P4HIR::ReferenceType>($_self).getObjectType()">/*,
                 DeclareOpInterfaceMethods<PromotableAllocationOpInterface>*/]> {
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
    %0 = p4hir.alloca !p4hir.bit<32> ["count", init] : !p4hir.ref<!p4hir.bit<32>>
    ...
    ```
  }];

  let arguments = (ins
    TypeAttr:$objectType,
    StrAttr:$name,
    UnitAttr:$init
  );

  let results = (outs Res<ReferenceType, ""/*,
                      [MemAlloc<AutomaticAllocationScopeResource>]*/>:$ref);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$ref, "mlir::Type":$objectType, "llvm::StringRef":$name)>,
  ];

  let assemblyFormat = [{
    $objectType ` `
    `[` $name
       (`,` `init` $init^)?
    `]`
     attr-dict `:` qualified(type($ref))
  }];

  let hasVerifier = 0;
}

def LoadOp : P4HIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches object type of 'ref'",
                 "ref", "result",
                 "mlir::cast<P4HIR::ReferenceType>($_self).getObjectType()">/*,
                 DeclareOpInterfaceMethods<PromotableMemOpInterface>*/]> {

  let summary = "Load value from variable";
  let description = [{
    `p4hir.load` reads a value given an object reference
    backed up by a `p4hir.ref` type.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = p4hir.load %0 : !p4hir.ref<!p4hir.bit<32>>, !p4hir.bit<32>

    ```
  }];

  let arguments = (ins Arg<ReferenceType, "the reference to load from"/*,
                           [MemRead]*/>:$ref);
  // FIXME: Constraint result type
  let results = (outs LoadableP4Type:$result);

  let assemblyFormat = [{
    $ref `:` qualified(type($ref)) `,` type($result) attr-dict
  }];

  // FIXME: add verifier.
}

def StoreOp : P4HIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches object type of 'addr'",
                 "ref", "value",
                 "mlir::cast<ReferenceType>($_self).getObjectType()">/*,
                 DeclareOpInterfaceMethods<PromotableMemOpInterface>*/]> {

  let summary = "Store value to variable";
  let description = [{
    `p4hir.store` stores a value (first operand) to the object referenced
    in the second operand.

    Example:

    ```mlir
    // Store a function argument to local variable, address in %0.
    p4hir.store %arg0, %0 : !p4hir.bit<32>, !p4hir.ref<!p4hir.bit<32>>
    ```
  }];

  let arguments = (ins LoadableP4Type:$value,
                       Arg<ReferenceType, "the object to store the value"/*,
                           [MemWrite]*/>:$ref);

  let assemblyFormat = [{
    $value `,` $ref attr-dict `:` type($value) `,` qualified(type($ref))
  }];

  // FIXME: add verifier.
}

// TODO: Decide if we'd want to be more precise and split cast into
// bitcast, trunc and extensions
def CastOp : P4HIR_Op<"cast",
             [Pure/*,
              DeclareOpInterfaceMethods<PromotableOpInterface>*/]> {
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

def UnaryOp : P4HIR_Op<"unary", [Pure, SameOperandsAndResultType]> {
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
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "Binary operations (arith and logic)";
  let description = [{
    p4hir.binop performs the binary operation according to
    the specified opcode kind: [mul, div, mod, add, sub,
    sadd, ssub, and, xor, or].

    It requires two input operands and has one result, all types
    should be the same.

    ```mlir
    %7 = p4hir.binop(add, %1, %2) : !p4hir.bit<32>
    %7 = p4hir.binop(mul, %1, %2) : !p4hir.bit<32>
    ```
  }];

  // TODO: get more accurate than AnyP4Type
  let results = (outs AnyP4Type:$result);
  let arguments = (ins Arg<BinOpKind, "binop kind">:$kind,
                       AnyP4Type:$lhs, AnyP4Type:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) attr-dict
  }];

  // TODO: Implement verification
  let hasVerifier = 0;
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

def CmpOp : P4HIR_Op<"cmp", [Pure, SameTypeOperands]> {
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
                 // "SwitchOp", "CaseOp",
                 // "ForInOp", "ForOp",
                 // "CallOp"
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

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
