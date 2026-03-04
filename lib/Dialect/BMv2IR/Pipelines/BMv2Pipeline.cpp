#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.h"
#include "p4mlir/Conversion/P4HIRToCoreLib/Passes.h"
#include "p4mlir/Dialect/BMv2IR/Pipelines/Passes.h"
#include "p4mlir/Transforms/Passes.h"

using namespace mlir;
using namespace P4::P4MLIR;

void P4::P4MLIR::buildBMv2Pipeline(OpPassManager &pm, const BMv2PipelineOpts &options) {
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createRemoveAliasesPass());
    pm.addPass(createFlattenStructsPass());

    // TODO: eliminate switches

    // Inlining passes
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(createInlineParsersPass());
    pm.addPass(createInlineControlsPass());

    // Eliminate temporaries
    pm.addPass(createCopyInCopyOutEliminationPass());

    // Lower to P4CoreLib
    pm.addPass(createSetCorelib());
    pm.addPass(createLowerToP4CoreLib());
    pm.addPass(createExpandEmitPass());

    // Lower to a more BMv2 friendly representation
    pm.addPass(createLowerToHeaderInstance());
    pm.addPass(createLowerPackage());
    pm.addPass(createSynthesizeActions());
    pm.addPass(createSynthesizeTables());
    pm.addPass(createUseControlPlaneNamesPass());

    // TODO: flatten structs and headers

    // Final lowering to BMv2IR
    pm.addPass(createP4HIRToBmv2IR());
    pm.addPass(createEnsureStandardMetadata());
    pm.addPass(createCanonicalizerPass());
}

void P4::P4MLIR::registerBMv2Pipeline() {
    PassPipelineRegistration<BMv2PipelineOpts>(
        "bmv2-pipeline", "Schedules middle-end passes for the BMv2 target", buildBMv2Pipeline);
}
