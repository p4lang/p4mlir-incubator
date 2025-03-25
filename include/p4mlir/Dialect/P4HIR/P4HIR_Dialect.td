#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// P4HIR dialect definition.
//===----------------------------------------------------------------------===//

def P4HIR_Dialect : Dialect {
    let name = "p4hir";
    let summary = "P4 High-level IR.";
    let description = [{
        P4HIR (P4 High-level IR) aims to represent a P4 source program at a
        high-level, and closely model the existing P4C IR.
    }];
    let cppNamespace = "::P4::P4MLIR::P4HIR";

    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 1;
    let hasConstantMaterializer = 1;

    let extraClassDeclaration = [{
        mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
        void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

        void registerAttributes();
        void registerTypes();
    }];
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_TD
