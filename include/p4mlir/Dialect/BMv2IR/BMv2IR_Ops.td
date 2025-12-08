#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"
include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.td"
include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.td"
include "p4mlir/Dialect/BMv2IR/BMv2IR_OpInterfaces.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"
include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.td"

//===----------------------------------------------------------------------===//
// Base BMv2IR operation definition.
//===----------------------------------------------------------------------===//

class BMv2IR_Op<string mnemonic, list<Trait> traits = []> :
        Op<BMv2IR_Dialect, mnemonic, traits>;

def BMv2IR_HeaderInstanceOp : BMv2IR_Op<"header_instance", [
    Symbol
  ]> {
  let summary = "Declares a header instance as a global variable";
  let description = [{
    Represents a header instance in a P4 program. Header instances
    are modeled similarly to global variables, and referenced by Symbol name by other ops.
    
    The operation has a symbol name and references a header type. It can be
    marked as metadata to distinguish metadata instances from regular header
    instances.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$header_type,
    DefaultValuedAttr<BoolAttr, "false">:$metadata
  );

  let assemblyFormat = [{
    $sym_name `:` $header_type attr-dict
  }];
}

def BMv2IR_SymToValueOp : BMv2IR_Op<"symbol_ref",
  [Pure, NoRegionArguments, AllowedParserOp,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {

  let results = (outs AnyType:$result);
  let arguments = (ins SymbolRefAttr:$decl);

  // TODO: Add verifier
  let hasVerifier = 0;

  let assemblyFormat = [{
    $decl attr-dict `:` type($result)
  }];
}

def BMv2IR_ParserOp : BMv2IR_Op<"parser", [
    Symbol, SymbolTable,
    SingleBlock, NoTerminator
  ]> {
  let summary = "P4 parser definition";
  let description = [{
    Represents a P4 parser with an initial state and a set of parse states.
    
    Example:
    ```mlir
    bmv2.parser @my_parser init_state @start {
      // parse states here
    }
    ```
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$init_state
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name `init_state` $init_state $body attr-dict
  }];
}

def BMv2IR_ParserStateOp : BMv2IR_Op<"state",
   [HasParent<"ParserOp">,
    Symbol, NoTerminator]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let description = [{ 
    Represents a BMv2 parse state.
    The op has 3 regions, which match the JSON format for parse states:
    - parser_ops contains the operations performed by the state to parse the header
    - transition_keys contains the keys from select transitions
    - transitions contains the actual transitions
  }];
  
  let regions = (region 
      MaxSizedRegion<1>:$parser_ops,
      MaxSizedRegion<1>:$transitions,
      MaxSizedRegion<1>:$transition_keys);
  
  let assemblyFormat = [{
    $sym_name
    `\n``transition_key` $transition_keys
    `\n``transitions` $transitions
    `\n``parser_ops` $parser_ops attr-dict
  }];

  let hasVerifier = 1;
}

def BMv2IR_TransitionOp : BMv2IR_Op<"transition", [HasParent<"ParserStateOp">]> {
  let summary = "Parser transition";
  let description = [{
    Represents a single transition from a parse state.
    
    Types:
    - default: default transition (no value/mask)
    - hexstr: value-based transition with hexstring
    - parse_vset: parse value-set based transition
    
    The value and mask are hexstrings in the format described in the BMv2 JSON
    spec (concatenation of byte-padded fields). For example, if the transition
    key has a 12-bit field and a 2-bit field, values use 3 bytes (0x0aba03).
    
    `next_state` can be null for the last state in the parser.
    `value` is the hexstr value or vset name (null if default)
    `mask` is the mask (if needed) for the hexstr
  }];
  
  // TODO: refine value and mask fields
  let arguments = (ins
    BMv2IR_TransitionKindAttr:$type,
    OptionalAttr<SymbolRefAttr>:$next_state,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<AnyAttr>:$mask
  );
  
  let assemblyFormat = "`type`$type (`value` $value^)? (`mask` $mask^)? (`next_state` $next_state^)? attr-dict";
}


def BMv2IR_ExtractOp : BMv2IR_Op<"extract", [HasParent<"ParserStateOp">, AllowedParserOp]> {
  let summary = "Extract header operation";
  let description = [{
    Extracts the field of a header instance, header stack, or union stack element.
    
    Types:
    - regular: extraction to a regular header instance
    - stack: extraction to the end of a header stack
    - union_stack: extraction to the end of a header union stack
  }];
  
  let arguments = (ins
    BMv2IR_ExtractKindAttr:$extract_type,
    SymbolRefAttr:$header_instance,
    OptionalAttr<StrAttr>:$union_member
  );
  
  let assemblyFormat = [{
    $extract_type $header_instance (`union_member` $union_member^)? attr-dict
  }];
}

def BMv2IR_AssignHeaderOp : BMv2IR_Op<"assign_header", [AllowedParserOp]> {
  let summary = "Assign header operation";
  let description = [{
    Assigns one header instance to another.
  }];

  
  let arguments = (ins
    SymbolRefAttr:$src,
    SymbolRefAttr:$dst
  );
  
  let assemblyFormat = [{
    $src `to` $dst attr-dict
  }];
}

def BMv2IR_AssignOp : BMv2IR_Op<"assign", [AllowedParserOp]> {
  let summary = "Assign operation";
  let description = [{
    Assigns the first argument to the second argument.
    Arguments can be arbitrary expressions.
  }];

  // TODO: refine types
  let arguments = (ins AnyType:$src,
                       AnyType:$dst);
  
  let assemblyFormat = [{
    $src `:` type($src) `to` $dst `:` type($dst) attr-dict
  }];
}

def BMv2IR_V1SwitchOp : BMv2IR_Op<"v1switch", [Symbol]> {
  let summary = "V1 Switch instantion";
  let description = [{
    Instantiates a V1 Switch package
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$parser,
    SymbolRefAttr:$verify_checksum,
    SymbolRefAttr:$ingress,
    SymbolRefAttr:$egress,
    SymbolRefAttr:$compute_checksum,
    SymbolRefAttr:$deparser
  );

  // TODO: add verifier to check that symbols have the correct type
  let hasVerifier = 1;
  let assemblyFormat = [{
    $sym_name `parser` $parser `,` `verify_checksum` $verify_checksum `,` `ingress` $ingress `,` `egress` $egress `,` `compute_checksum` $compute_checksum `,` `deparser` $deparser attr-dict
  }];
}
    
def BMv2IR_DataToBoolOp : BMv2IR_Op<"d2b"> {
  let summary = "Data to Boolean operation";
  let description = [{
    Unary operation which can be used to explicitly convert a data value (i.e. a value that can
    be read from / written to a field, or a value obtained from runtime action data,
    or a value obtained from a JSON hexstr) to a boolean value.
  }];

  let arguments = (ins BMv2IR_FieldType:$input);
  let results = (outs BooleanType:$result);
  let assemblyFormat = [{
    $input `:` type($input) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      build($_builder, $_state, ::P4::P4MLIR::P4HIR::BoolType::get($_state.getContext()), input);
    }]>
  ];
}

//===-------------------------------------------------------------------------------------------===//
// BMv2 type-value fields
// see https://github.com/p4lang/behavioral-model/blob/main/docs/JSON_format.md#the-type-value-object
//===-------------------------------------------------------------------------------------------===//

def BMv2IR_FieldOp : BMv2IR_Op<"field", [AllowedTransitionKey, AllowedParserOp]> {
  let summary = "BMv2 header field access";
  let description = [{
    Represents access to a field within a header instance.
    Since in BMv2 header instances can contain only bits, this op can return only bits.
    Contains a 2-tuple: (header_instance_name, field_member_name)
  }];
  
  let arguments = (ins
    SymbolRefAttr:$headerInstance,
    StrAttr:$fieldMember
  );
  
  let results = (outs 
    BMv2IR_FieldType:$res); 
  
  let assemblyFormat = "$headerInstance `[` $fieldMember `]` `->` type($res) attr-dict";
}

def BMv2_StackFieldOp : BMv2IR_Op<"stack_field", [AllowedTransitionKey]> {
  let summary = "BMv2 header stack field access";
  let description = [{
    Represents access to a field in the last valid header instance in a stack.
    Contains a 2-tuple: (header_stack_name, field_member_name)
  }];
  
  let arguments = (ins
    StrAttr:$headerStack,
    StrAttr:$fieldMember
  );
  
  let assemblyFormat = "`<` $headerStack `,` $fieldMember `>` attr-dict";
}


def BMv2IR_LookaheadOp : BMv2IR_Op<"lookahead", [AllowedTransitionKey]> {
  let summary = "BMv2 parser lookahead";
  let description = [{
    Represents a lookahead operation in a parser.
    Contains a 2-tuple: (bit_offset, bitwidth)
  }];
  
  let arguments = (ins
    I32Attr:$bitOffset,
    I32Attr:$bitwidth
  );
  
  let assemblyFormat = "`<` $bitOffset `,` $bitwidth `>` attr-dict";
}

def BMv2_UnionStackFieldOp : BMv2IR_Op<"union_stack_field", [AllowedTransitionKey]> {
  let summary = "BMv2 header union stack field access";
  let description = [{
    Represents access to a field in the last valid union instance in a stack.
    Contains a 3-tuple: (header_union_stack_name, union_member_name, field_member_name)
  }];
  
  let arguments = (ins
    StrAttr:$headerUnionStack,
    StrAttr:$unionMember,
    StrAttr:$fieldMember
  );
  
  let assemblyFormat = "`<` $headerUnionStack `,` $unionMember `,` $fieldMember `>` attr-dict";
}

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD
