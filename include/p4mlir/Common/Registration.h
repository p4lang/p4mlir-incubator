#ifndef P4MLIR_COMMON_REGISTRATION_H
#define P4MLIR_COMMON_REGISTRATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/DialectRegistry.h"
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.h"
#include "p4mlir/Conversion/P4HIRToCoreLib/Passes.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/Pipelines/Passes.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/Pipelines/Passes.h"
#include "p4mlir/Targets/BMv2/Target.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
    registry.insert<P4::P4MLIR::P4HIR::P4HIRDialect, P4::P4MLIR::P4CoreLib::P4CoreLibDialect,
                    P4::P4MLIR::BMv2IR::BMv2IRDialect, mlir::func::FuncDialect,
                    mlir::pdl::PDLDialect, mlir::pdl_interp::PDLInterpDialect>();
}

inline void registerAllPassesAndPipelines() {
    P4::P4MLIR::registerPasses();
    P4::P4MLIR::registerP4HIRToCoreLibPasses();
    P4::P4MLIR::registerP4HIRToBMv2IRPasses();
    P4::P4MLIR::registerCommonFrontEndPipeline();
    P4::P4MLIR::registerBMv2Pipeline();
    P4::P4MLIR::registerP4HIRStrengthReductionPass();
}

inline void registerAllTranslations() { P4::P4MLIR::registerToBMv2JSONTranslation(); }
}  // namespace P4::P4MLIR

#endif  // P4MLIR_COMMON_REGISTRATION_H
