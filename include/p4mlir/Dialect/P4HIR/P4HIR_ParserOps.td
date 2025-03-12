#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PARSEROPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PARSEROPS_TD

def ParserOp : P4HIR_Op<"parser",
    [Symbol, SymbolTable,
     FunctionOpInterface, AutomaticAllocationScope]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FuncType>:$applyType,
                       TypeAttrOf<CtorType>:$ctorType);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$sym_name, "P4HIR::FuncType":$applyType, "P4HIR::CtorType":$ctorType)>
  ];

  let extraClassDeclaration = [{
    mlir::Region *getCallableRegion() { return &getBody(); }

    auto getFunctionType() { return getApplyType(); }

    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getApplyType().getInputs();
    }

    void setFunctionTypeAttr(mlir::TypeAttr attr) {
      getProperties().applyType = attr;
    }

    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return {};
    }

    void createEntryBlock();
  }];
}

def ParserStateOp : P4HIR_Op<"state",
   [Symbol, AutomaticAllocationScope]> {
 let arguments = (ins SymbolNameAttr:$sym_name);
 let regions = (region MinSizedRegion<1>:$body);
 let assemblyFormat = [{
   $sym_name $body attr-dict
 }];
}

def ParserTransitionOp : P4HIR_Op<"transition",
   [Terminator,
    ParentOneOf<["ParserStateOp", "ParserOp"]>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins SymbolRefAttr:$state);
  let assemblyFormat = [{
    `to` $state attr-dict
  }];
}

def ParserAcceptOp : P4HIR_Op<"parser_accept",
   [Terminator,
    ParentOneOf<["ParserStateOp"]>]> {
  let assemblyFormat = "attr-dict";
}

def ParserRejectOp : P4HIR_Op<"parser_reject",
   [Terminator,
    ParentOneOf<["ParserStateOp",
                 "IfOp" // for verify() lowering
               ]>]> {
  let arguments = (ins OptionalAttr<P4HIR_ErrorCodeAttr>:$error);
  let assemblyFormat = "attr-dict (`with` `error` $error^)?";

  let builders = [
    // Convenience builder without an error
    OpBuilder<(ins), [{
      build($_builder, $_state, {});
    }]>
  ];
}

def ParserTransitionSelectOp : P4HIR_Op<"transition_select",
   [Terminator, NoTerminator,
    ParentOneOf<["ParserStateOp"]>]> {
  // FIXME: constraint argument better. Should be tuple or "normal" type.
  let arguments = (ins AnyP4Type:$select);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $select `:` type($select) $body attr-dict
  }];
  // FIXME: Implement verifier for cases
}

def ParserSelectCaseOp : P4HIR_Op<"select_case",
   [AutomaticAllocationScope,
    ParentOneOf<["ParserTransitionSelectOp"]>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins SymbolRefAttr:$state);
  let regions = (region MinSizedRegion<1>:$key);
  let assemblyFormat = [{
     $key `to` $state attr-dict
  }];
  // TBD: check region
  let hasVerifier = 0;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$keyBuilder,
                   "llvm::StringRef":$nextState)>,
    OpBuilder<(ins "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$keyBuilder,
                   "mlir::SymbolRefAttr":$nextState)>
  ];
}

def SetOp : P4HIR_Op<"set",
   [Pure, SameTypeOperands,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a set from constituent parts.";
  // FIXME: Better constraint type
  let arguments = (ins Variadic<AnyP4Type>:$input);
  let results = (outs SetType:$result);
  let hasCustomAssemblyFormat = 1;
  // FIXME: use declarative format
  // let assemblyFormat = [{
  //   `(` $input `)` attr-dict `:` type($result)
  // }];

  // TODO: Automatically infer result type
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$input)>
  ];

  let hasVerifier = 1;
}

def SetProductOp : P4HIR_Op<"set_product",
   [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a carthesian set product";
  // FIXME: Better constraint type
  let arguments = (ins Variadic<SetType>:$input);
  let results = (outs SetType:$result);
  let hasCustomAssemblyFormat = 1;
  // FIXME: use declarative format
  // let assemblyFormat = [{
  //   `(` $input `)` attr-dict `:` type($result)
  // }];

  // TODO: Automatically infer result type
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$input)>
  ];

  let hasVerifier = 1;
}

def UniversalSetOp : P4HIR_Op<"universal_set",
   [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create a universal set";
  // TODO: Make it properly types
  // let arguments = (ins AnyP4Type:$input);
  let results = (outs SetType:$result);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins)>
  ];
}

def RangeOp : P4HIR_Op<"range", [Pure,
  TypesMatchWith<
       "the resulting type of a range operation must be the set of left-hand side operand types",
       "result", "lhs", "cast<P4HIR::SetType>($_self).getElementType()">,
  TypesMatchWith<
       "the resulting type of a range operation must be the set of right-hand side operand types",
       "result", "rhs", "cast<P4HIR::SetType>($_self).getElementType()">,
  ParentOneOf<["ParserSelectCaseOp"]>, // TBD: ForInOp
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

  let summary = "Range operation";
  let description = [{
    p4hir.range represents a set formed by all values numerically between the first and the second, inclusively.

    It requires two input operands and has one result, with input operands
    being of any integer type.

    ```mlir
    %3 = p4hir.range(%1, %2) : !p4hir.set<!p4hir.bit<42>>
    ```
  }];

  let results = (outs SetType:$result);
  let arguments = (ins AnyIntP4Type:$lhs, AnyIntP4Type:$rhs);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` `:` qualified(type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      auto resultType = SetType::get($_builder.getContext(), lhs.getType());
      build($_builder, $_state, resultType, lhs, rhs);
    }]>
  ];
}

def MaskOp : P4HIR_Op<"mask", [Pure,
  TypesMatchWith<
       "the resulting type of a mask operation must be the set of left-hand side operand types",
       "result", "lhs", "cast<P4HIR::SetType>($_self).getElementType()">,
  TypesMatchWith<
       "the resulting type of a mask operation must be the set of right-hand side operand types",
       "result", "rhs", "cast<P4HIR::SetType>($_self).getElementType()">,
  ParentOneOf<["ParserSelectCaseOp"]>,       
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

  let summary = "Mask operation";
  let description = [{
    p4hir.mask represents a set formed by all values defined by the mask.
    More formally, the set denoted by a &&& b is defined as follows:
    ```
    a &&& b = { c where a & b = c & b }
    ```

    It requires two input operands and has one result, with input operands
    being of any integer type.

    ```mlir
    %3 = p4hir.mask(%1, %2) : !p4hir.set<!p4hir.bit<42>>
    ```
  }];

  let results = (outs SetType:$result);
  let arguments = (ins AnyIntP4Type:$lhs, AnyIntP4Type:$rhs);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` `:` qualified(type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      auto resultType = SetType::get($_builder.getContext(), lhs.getType());
      build($_builder, $_state, resultType, lhs, rhs);
    }]>
  ];
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PARSEROPS_TD
