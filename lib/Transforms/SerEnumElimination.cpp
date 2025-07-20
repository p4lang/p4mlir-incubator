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
    SerEnumEliminationPass() = default;
    void runOnOperation() override;
};

class SerEnumTypeConverter : public P4HIRTypeConverter {
 public:
    explicit SerEnumTypeConverter() {
        // Converts SerEnumType to the underlying BitsType
        addConversion([](P4HIR::SerEnumType serEnumType) { return serEnumType.getType(); });

        addTypeAttributeConversion([](P4HIR::SerEnumType serEnumType, P4HIR::EnumFieldAttr attr) {
            return serEnumType.valueOf<P4HIR::IntAttr>(attr.getField());
        });
    }
};
}  // namespace

void SerEnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    SerEnumTypeConverter typeConverter;
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<P4HIR::OverloadSetOp>([&](P4HIR::OverloadSetOp ovl) {
        for (auto opFunc : ovl.getOps<P4HIR::FuncOp>()) return !target.isIllegal(opFunc);
        return true;
    });

    target.addDynamicallyLegalOp<P4HIR::CaseOp>([&](P4HIR::CaseOp caseOp) {
        return llvm::all_of(caseOp.getValue(), [&](Attribute val) {
            if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(val)) {
                return typeConverter.isLegal(typedAttr.getType());
            }
            return true;
        });
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (auto func = dyn_cast<FunctionOpInterface>(op)) {
            auto fnType = func.getFunctionType();
            return typeConverter.isLegal(fnType);
        }

        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(&context);

    // Translate call operands and results via type converter
    populateP4HIRAnyCallOpTypeConversionPattern(patterns, typeConverter);
    // Translate function-like ops signatures and types
    populateP4HIRFunctionOpTypeConversionPattern<P4HIR::FuncOp, P4HIR::ParserOp, P4HIR::ControlOp>(
        patterns, typeConverter);
    patterns.add<TypeConversionPattern>(typeConverter, &context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createSerEnumEliminationPass() {
    return std::make_unique<SerEnumEliminationPass>();
}
