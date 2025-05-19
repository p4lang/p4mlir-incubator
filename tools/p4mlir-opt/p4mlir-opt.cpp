#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
    mlir::registerTransformsPasses();

    P4::P4MLIR::registerPasses();
    P4::P4MLIR::registerP4MLIRConversionPasses();

    mlir::DialectRegistry registry;
    registry.insert<P4::P4MLIR::P4HIR::P4HIRDialect, P4::P4MLIR::P4CoreLib::P4CoreLibDialect,
                    mlir::func::FuncDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "P4MLIR optimizer driver\n", registry));
}
