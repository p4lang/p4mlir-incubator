#ifndef P4MLIR_DIALECT_BMv2IR_BMv2IR_DIALECT_TD
#define P4MLIR_DIALECT_BMv2IR_BMv2IR_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BMv2IR dialect definition.
//===----------------------------------------------------------------------===//

def BMv2IR_Dialect : Dialect {
    let name = "bmv2ir";
    let summary = "BMv2 IR";
    let description = [{
        BMv2IR is an MLIR dialect that closely matches the
        [BMv2 JSON format](https://github.com/p4lang/behavioral-model/blob/main/docs/JSON_format.md),
        actiong as an intermediate for between higher level dialects (P4HIR, P4CoreLib) and the actual
        JSON representation of the program.
    }];
    let cppNamespace = "::P4::P4MLIR::BMv2IR";

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();
    }];
}

#endif // P4MLIR_DIALECT_BMv2IR_BMv2IR_DIALECT_TD
