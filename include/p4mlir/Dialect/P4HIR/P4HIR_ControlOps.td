#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_CONTROLOPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_CONTROLOPS_TD

def ControlOp : P4HIR_Op<"control",
    [Symbol, SymbolTable,
     FunctionOpInterface, AutomaticAllocationScope]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FuncType>:$applyType,
                       TypeAttrOf<CtorType>:$ctorType,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$sym_name, "P4HIR::FuncType":$applyType, "P4HIR::CtorType":$ctorType,
               CArg<"llvm::ArrayRef<mlir::DictionaryAttr>", "llvm::ArrayRef<mlir::DictionaryAttr>()">:$argAttrs,
               CArg<"mlir::DictionaryAttr", "mlir::DictionaryAttr()">:$annotations)>
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

def ControlApplyOp : P4HIR_Op<"control_apply",
   [AutomaticAllocationScope, NoRegionArguments,
    Terminator, NoTerminator,
    ParentOneOf<["ControlOp"]>,]> {
 let regions = (region MaxSizedRegion<1>:$body);
 let assemblyFormat = [{
   $body attr-dict
 }];
}

def TableOp : P4HIR_Op<"table",
   [Symbol, NoTerminator,
    ParentOneOf<["ControlOp"]>]> {
  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::DictionaryAttr":$annotations,
              "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$entryBuilder)>,
  ];

  let assemblyFormat = [{
    $sym_name (`annotations` $annotations^)? $body attr-dict
  }];
}

def TableApplyOp : P4HIR_Op<"table_apply",
  [NoRegionArguments, CallOpInterface,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
   ]> {
  let summary = "table apply operation";
  let description = [{
    Applies P4 table

    Example:

    ```mlir
    ...
    ```
  }];

  let results = (outs StructType:$apply_result);
  let arguments = (ins SymbolRefAttr:$callee);

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

    mlir::OperandRange getArgOperands() {
      return { getOperation()->operand_end(), getOperation()->operand_end() };
    }

    mlir::MutableOperandRange getArgOperandsMutable() {
      llvm_unreachable("no operands for table apply");
    }
  }];

  let assemblyFormat = [{
    $callee attr-dict `:` type($apply_result)
  }];
}


def TableEntryOp : P4HIR_Op<"table_entry", [
   NoRegionArguments, SingleBlockImplicitTerminator<"YieldOp">,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
   ]> {
  let summary = "Represents a named P4 table entry";
  let description = [{

  }];

  // TODO: Refine result type, need to allow externs as well as normal P4 types
  let results = (outs Optional<AnyType>:$result);
  let arguments = (ins StrAttr:$name, UnitAttr:$is_const, OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);

  // TODO: Check
  let hasVerifier = 0;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    $name (`const` $is_const^)? (`annotations` $annotations^)? $body (`:` type($result)^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$name, "bool":$is_const, "mlir::DictionaryAttr":$annotations,
              "llvm::function_ref<void(mlir::OpBuilder &, mlir::Type &, mlir::Location)>":$entryBuilder)>,
  ];
}

def TableActionsOp : P4HIR_Op<"table_actions", [
       NoRegionArguments, NoTerminator]> {
  let summary = "Represents a list of actions for a P4 table";
  let description = [{

  }];

  let arguments = (ins OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);

  // TODO: Check
  let hasVerifier = 0;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    (`annotations` $annotations^)? $body attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::DictionaryAttr":$annotations,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$entryBuilder)>,
  ];
}

def TableActionOp : P4HIR_Op<"table_action",
    [AutomaticAllocationScope, NoTerminator,
     // DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let arguments = (ins TypeAttrOf<FuncType>:$cplaneType, SymbolRefAttr:$action,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$action, "P4HIR::FuncType":$cplaneType,
                   "llvm::ArrayRef<mlir::DictionaryAttr>":$argAttrs,
                   "mlir::DictionaryAttr":$annotations,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Block::BlockArgListType, mlir::Location)>":$entryBuilder)>
  ];
}


def TableDefaultActionOp : P4HIR_Op<"table_default_action",
  [NoRegionArguments, NoTerminator]> {
  let summary = "Represents a P4 table default action";
  let description = [{

  }];

  let arguments = (ins OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);

  // TODO: Check
  let hasVerifier = 0;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    (`annotations` $annotations^)? $body attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::DictionaryAttr":$annotations,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$entryBuilder)>,
  ];
}

def TableSizeOp : P4HIR_Op<"table_size",
  [ConstantLike, Pure,
   AllTypesMatch<["value", "result"]>,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>] > {
  let summary = "Represents a table size for a P4 table";
  let description = [{

  }];

  let results = (outs AnyIntP4Type:$result);
  let arguments = (ins TypedAttrInterface:$value, OptionalAttr<DictionaryAttr>:$annotations);

  let assemblyFormat = [{
    $value (`annotations` $annotations^)? attr-dict
  }];
}

def TableKeyOp : P4HIR_Op<"table_key",
  [NoRegionArguments, NoTerminator]> {
  let summary = "Represents a P4 table key";
  let description = [{

  }];

  let arguments = (ins OptionalAttr<DictionaryAttr>:$annotations);
  let regions = (region SizedRegion<1>:$body);

  // TODO: Check
  let hasVerifier = 0;
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    (`annotations` $annotations^)? $body attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::DictionaryAttr":$annotations,
                   "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>":$keyBuilder)>,
  ];
}

def TableKeyEntryOp : P4HIR_Op<"match_key", []> {
  let summary = "Represents a P4 table key entry";
  let description = [{

  }];

  let arguments = (ins P4HIR_MatchKindAttr:$match_kind, AnyP4Type:$value,
                       OptionalAttr<DictionaryAttr>:$annotations);

  // TODO: Check
  let hasVerifier = 0;
  let assemblyFormat = [{
    $match_kind $value `:` type($value) (`annotations` $annotations^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$match_kind, "mlir::Value":$value,
                   "mlir::DictionaryAttr":$annotations), [{
      build($_builder, $_state,
            P4HIR::MatchKindAttr::get($_builder.getContext(), match_kind), value,
            annotations.empty() ? mlir::DictionaryAttr() : annotations);
    }]
    >,
  ];

}

def ExitOp : P4HIR_Op<"exit", [ParentOneOf<["ScopeOp", "IfOp",
                                            "FuncOp", "ControlApplyOp",]>]> {
  let summary = "Represents an exit from action or control";
  let description = [{

  }];

  // TODO: Check that it is invoked only inside action
  let hasVerifier = 0;

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = [{ attr-dict }];
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_CONTROLOPS_TD
