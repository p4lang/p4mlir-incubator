#ifndef P4MLIR_DIALECT_BMv2IR_PIPELINES_PASSES_H
#define P4MLIR_DIALECT_BMv2IR_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace P4::P4MLIR {

struct BMv2PipelineOpts : public mlir::PassPipelineOptions<BMv2PipelineOpts> {};

/// Adds common FrontEnd pass to the pass manager
void buildBMv2Pipeline(mlir::OpPassManager &pm, const BMv2PipelineOpts &options);

void registerBMv2Pipeline();

}  // namespace P4::P4MLIR

#endif  // P4MLIR_DIALECT_BMv2IR_PIPELINES_PASSES_H
