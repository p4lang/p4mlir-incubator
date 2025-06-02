#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.td"
include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

//===----------------------------------------------------------------------===//
// Base P4CoreLob operation definition.
//===----------------------------------------------------------------------===//

class P4CoreLib_Op<string mnemonic, list<Trait> traits = []> :
        Op<P4CoreLib_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// P4CoreLib operation definitions.
//===----------------------------------------------------------------------===//

def VerifyOp : P4CoreLib_Op<"verify",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>]> {
    let summary = "Check given predicate in the parser";
    let description = [{
        Check a predicate @check in the parser; if the predicate is true do nothing,
        otherwise set the parser error to @toSignal, and transition to the `reject` state.
    }];

    let arguments = (ins BooleanType:$check, ErrorType:$toSignal);
    let assemblyFormat = [{
      $check `signalling` $toSignal `:` type($toSignal) attr-dict
    }];
}

def PacketEmitOp : P4CoreLib_Op<"emit",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ControlApplyOp"]>]> {
    let summary = "Write header-like object to output packet";
    let description = [{
        Write @hdr into the output packet, advancing cursor.
        @hdr can be of a header type, a header stack, a header_union, or a struct
        containing fields with such types.
    }];

    let arguments = (ins PacketOutType:$packet_out, StructLikeType:$hdr);
    let assemblyFormat = [{
      $hdr `:` type($hdr) `to` $packet_out `:` type($packet_out) attr-dict
    }];
}

def PacketLengthOp : P4CoreLib_Op<"packet_length",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

    let arguments = (ins PacketInType:$packet_in);
    let results = (outs Bit<32>:$length);
    let assemblyFormat = [{
      $packet_in `:` type($packet_in) attr-dict `->` type($length)
    }];
}

def PacketLookAheadOp : P4CoreLib_Op<"packet_lookahead",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {

    let arguments = (ins PacketInType:$packet_in);
    let results = (outs AnyType:$value);
    let assemblyFormat = [{
      $packet_in `:` type($packet_in) attr-dict `->` type($value)
    }];
}

def PacketAdvanceOp : P4CoreLib_Op<"packet_advance",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>]> {

    let arguments = (ins PacketInType:$packet_in, Bit<32>:$bits);
    let assemblyFormat = [{
      $packet_in `:` type($packet_in) `by` $bits attr-dict
    }];
}

def PacketExtractOp : P4CoreLib_Op<"extract_header",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>]> {

    let arguments = (ins PacketInType:$packet_in, HeaderRefType:$hdr);
    let assemblyFormat = [{
      $hdr `:` type($hdr) `from` $packet_in `:` type($packet_in) attr-dict
    }];
}

def PacketExtractVariableOp : P4CoreLib_Op<"extract_header_variable",
    [ParentOneOf<["P4::P4MLIR::P4HIR::ParserStateOp"]>]> {

    let arguments = (ins PacketInType:$packet_in, HeaderRefType:$hdr, Bit<32>:$variableFieldSizeInBits);
    let assemblyFormat = [{
      $hdr`<`$variableFieldSizeInBits`>`  `:` type($hdr) `from` $packet_in `:` type($packet_in) attr-dict
    }];
}

#endif //P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_TD
