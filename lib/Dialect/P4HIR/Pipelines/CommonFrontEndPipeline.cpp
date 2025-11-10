#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Dialect/P4HIR/Pipelines/Passes.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Transforms/Passes.h"

using namespace mlir;
using namespace P4::P4MLIR;

void P4::P4MLIR::buildCommonFrontEndPassPipeline(OpPassManager &pm,
                                                 const CommonFrontEndPipelineOpts &options) {
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(createRemoveSoftCFPass());

    pm.addPass(createLowerToP4CoreLib());
    // TODO: add target-specific extern conversion pass with an appropriate hook
    // to define target-specific conversions

    pm.addPass(createCopyInCopyOutEliminationPass());

    // TODO: add Symbol DCE/CSE passes
}

void P4::P4MLIR::registerCommonFrontEndPipeline() {
    PassPipelineRegistration<CommonFrontEndPipelineOpts>(
        "p4-front-end-pipeline", "Schedules front end passes that are common among every P4 target",
        buildCommonFrontEndPassPipeline);
}
