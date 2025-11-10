// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-ser-enum-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SERENUMELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct SerEnumEliminationPass
    : public P4::P4MLIR::impl::SerEnumEliminationBase<SerEnumEliminationPass> {
    void runOnOperation() override;
};

void SerEnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    P4HIRTypeConverter converter;
    converter.addConversion([](P4HIR::SerEnumType serEnumType) { return serEnumType.getType(); });

    converter.addTypeAttributeConversion(
        [](P4HIR::SerEnumType serEnumType, P4HIR::EnumFieldAttr attr) {
            return serEnumType.valueOf<P4HIR::IntAttr>(attr.getField());
        });

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    populateTypeConversionPattern(patterns, converter);
    configureUnknownOpDynamicallyLegalByTypes(target, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createSerEnumEliminationPass() {
    return std::make_unique<SerEnumEliminationPass>();
}
