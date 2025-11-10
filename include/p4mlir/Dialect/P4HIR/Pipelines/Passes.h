#ifndef P4MLIR_DIALECT_P4HIR_PIPELINES_PASSES_H
#define P4MLIR_DIALECT_P4HIR_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace P4::P4MLIR {

struct CommonFrontEndPipelineOpts : public mlir::PassPipelineOptions<CommonFrontEndPipelineOpts> {};

/// Adds common FrontEnd pass to the pass manager
void buildCommonFrontEndPassPipeline(mlir::OpPassManager &pm,
                                     const CommonFrontEndPipelineOpts &options);

void registerCommonFrontEndPipeline();

}  // namespace P4::P4MLIR

#endif  // P4MLIR_DIALECT_P4HIR_PIPELINES_PASSES_H
