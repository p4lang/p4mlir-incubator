#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_TD
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.td"

//===----------------------------------------------------------------------===//
// P4CoreLib type definitions.
//===----------------------------------------------------------------------===//

class P4CoreLib_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4CoreLib_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PacketInType : P4CoreLib_Type<"PacketIn", "packet_in"> {
  let summary = "packet_in extern";
  let description = [{
    Represents packet_in extern
  }];
}

def PacketOutType : P4CoreLib_Type<"PacketOut", "packet_out"> {
  let summary = "packet_out extern";
  let description = [{
    Represents packet_out extern
  }];
}

#endif // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_TYPES_TD
