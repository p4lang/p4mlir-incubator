#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD

include "mlir/IR/OpBase.td"

include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.td"
include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.td"

//===----------------------------------------------------------------------===//
// Base BMv2IR operation definition.
//===----------------------------------------------------------------------===//

class BMv2IR_Op<string mnemonic, list<Trait> traits = []> :
        Op<BMv2IR_Dialect, mnemonic, traits>;

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_OPS_TD
