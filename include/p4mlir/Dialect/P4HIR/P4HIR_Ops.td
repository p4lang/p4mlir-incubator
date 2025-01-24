#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
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

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
