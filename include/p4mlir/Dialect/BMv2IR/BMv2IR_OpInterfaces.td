#ifndef BMv2IR_OP_INTERFACES_TD
#define BMv2IR_OP_INTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"

def AllowedTransitionKey : OpInterface<"AllowedTransitionKey"> {
  let description = [{
    Common interfaces for operations that could be used as transition keys
  }];
  let methods = [];


  let cppNamespace = "::P4::P4MLIR::BMv2IR";
}

def AllowedParserOp : OpInterface<"AllowedParserOp"> {
  let description = [{
    Common interfaces for operations that could be used as parser ops
  }];
  let methods = [];


  let cppNamespace = "::P4::P4MLIR::BMv2IR";
}

#endif
