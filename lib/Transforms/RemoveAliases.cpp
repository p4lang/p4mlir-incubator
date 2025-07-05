#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-alias"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVEALIASES
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveAliasesPass : public P4::P4MLIR::impl::RemoveAliasesBase<RemoveAliasesPass> {
    RemoveAliasesPass() = default;
    void runOnOperation() override;
};

class AliasTypeConverter : public P4HIRTypeConverter {
 public:
    AliasTypeConverter(MLIRContext *context) {
        addConversion([&](P4HIR::AliasType aliasType) -> Type {
            return convertType(aliasType.getAliasedType());
        });

        addTypeAttributeConversion([&](P4HIR::AliasType aliasType, Attribute attr) -> Attribute {
            Type underlyingType = convertType(aliasType);

            return llvm::TypeSwitch<Attribute, Attribute>(attr)
                .Case<P4HIR::IntAttr>([&](P4HIR::IntAttr intAttr) {
                    return P4HIR::IntAttr::get(underlyingType, intAttr.getValue());
                })
                .Case<P4HIR::BoolAttr>([&](P4HIR::BoolAttr boolAttr) {
                    return P4HIR::BoolAttr::get(context, boolAttr.getValue());
                })
                .Default([](Attribute) -> Attribute {
                    llvm_unreachable("Unsupported attribute type for alias conversion");
                });
        });
    }
};

}  // namespace

void RemoveAliasesPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    AliasTypeConverter typeConverter(&context);
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<P4HIR::OverloadSetOp>([&](P4HIR::OverloadSetOp ovl) {
        for (auto opFunc : ovl.getOps<P4HIR::FuncOp>()) return !target.isIllegal(opFunc);
        return true;
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

std::unique_ptr<Pass> P4::P4MLIR::createRemoveAliasesPass() {
    return std::make_unique<RemoveAliasesPass>();
}
