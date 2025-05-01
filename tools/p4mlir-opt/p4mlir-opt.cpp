#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    P4::P4MLIR::registerPasses();

    mlir::DialectRegistry registry;
    registry.insert<P4::P4MLIR::P4HIR::P4HIRDialect, mlir::func::FuncDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "P4MLIR optimizer driver\n", registry));
}
