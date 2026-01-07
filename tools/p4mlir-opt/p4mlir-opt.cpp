#include "p4mlir/Dialect/BMv2/BMv2Dialect.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Common/Registration.h"

int main(int argc, char **argv) {
    mlir::registerTransformsPasses();

    P4::P4MLIR::registerAllPassesAndPipelines();

    mlir::DialectRegistry registry;
    P4::P4MLIR::registerAllDialects(registry);
    registry.insert<p4mlir::bmv2::BMv2Dialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "P4MLIR optimizer driver\n", registry));
}