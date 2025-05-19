#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_TD
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// P4CoreLib dialect definition.
//===----------------------------------------------------------------------===//

def P4CoreLib_Dialect : Dialect {
    let name = "p4corelib";
    let summary = "P4-16 Core Library IR";
    let description = [{
        Dialect used to represent operations defined in P4-16 core library (core.p4).
    }];
    let cppNamespace = "::P4::P4MLIR::P4CoreLib";

    let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
    ];
}

#endif // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_DIALECT_TD
