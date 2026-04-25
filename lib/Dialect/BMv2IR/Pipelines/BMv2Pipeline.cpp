#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Dialect/BMv2IR/Pipelines/Passes.h"
#include "p4mlir/Transforms/Passes.h"

using namespace mlir;
using namespace P4::P4MLIR;

void P4::P4MLIR::buildBMv2Pipeline(OpPassManager &pm, const BMv2PipelineOpts &options) {
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createEnumEliminationPass());
    pm.addPass(createSerEnumEliminationPass());
    pm.addPass(createRemoveAliasesPass());
    pm.addPass(createFlattenStructsPass());

    // TODO: eliminate switches

    pm.addPass(mlir::createInlinerPass());
    // TODO: add parsers and controls inlining once #246 is merged

    // TODO: implement store to load forwarding

    // TODO: flatten structs and headers

    // TODO: implement action synthesis,

    // TODO: implement action call to table conversion
}

void P4::P4MLIR::registerBMv2Pipeline() {
    PassPipelineRegistration<BMv2PipelineOpts>(
        "bmv2-pipeline", "Schedules middle-end passes for the BMv2 target", buildBMv2Pipeline);
}
