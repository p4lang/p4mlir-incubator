// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/P4HIRToCoreLib/ConversionPatterns.h"
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
    void runOnOperation() override;
};

void RemoveAliasesPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();
    P4HIRTypeConverter converter;

    converter.addConversion([&](P4HIR::AliasType aliasType) {
        return converter.convertType(aliasType.getAliasedType());
    });

    converter.addTypeAttributeConversion([&](P4HIR::AliasType aliasType, Attribute attr) {
        Type underlyingType = converter.convertType(aliasType);

        return llvm::TypeSwitch<Attribute, Attribute>(attr)
            .Case<P4HIR::IntAttr>([&](P4HIR::IntAttr intAttr) {
                return P4HIR::IntAttr::get(underlyingType, intAttr.getValue());
            })
            .Case<P4HIR::BoolAttr>([&](P4HIR::BoolAttr boolAttr) {
                return P4HIR::BoolAttr::get(attr.getContext(), boolAttr.getValue());
            })
            .Default([](Attribute) -> Attribute {
                llvm_unreachable("Unsupported attribute type for alias conversion");
            });
    });

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    populateTypeConversionPattern(patterns, converter);
    configureUnknownOpDynamicallyLegalByTypes(target, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createRemoveAliasesPass() {
    return std::make_unique<RemoveAliasesPass>();
}
