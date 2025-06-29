#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/DialectConversion.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-enum-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_ENUMELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct EnumEliminationPass : public P4::P4MLIR::impl::EnumEliminationBase<EnumEliminationPass> {
    EnumEliminationPass() = default;
    void runOnOperation() override;
};
}  // namespace

class EnumTypeConverter : public TypeConverter {
 public:
    explicit EnumTypeConverter(MLIRContext *ctx) {
        // Fallback for other types.
        addConversion([](Type type) -> Type { return type; });

        // Converts EnumType to SerEnumType
        addConversion([ctx](P4HIR::EnumType enumType) -> Type {
            auto underlyingType = mlir::cast<P4HIR::BitsType>(
                mlir::cast<P4HIR::EnumRepresentationInterface>(enumType).getUnderlyingType());

            llvm::SmallVector<mlir::NamedAttribute> fields;
            for (auto const &[index, field] : llvm::enumerate(enumType.getFields())) {
                auto name = mlir::cast<StringAttr>(field);
                auto value = P4HIR::IntAttr::get(ctx, underlyingType,
                                                 APInt(underlyingType.getWidth(), index));
                fields.emplace_back(name, value);
            }
            return P4HIR::SerEnumType::get(enumType.getName(), underlyingType, fields,
                                           enumType.getAnnotations());
        });

        addConversion([this](P4HIR::ReferenceType refType) -> std::optional<Type> {
            auto newType = convertType(refType.getObjectType());
            if (!newType) return std::nullopt;

            return P4HIR::ReferenceType::get(newType);
        });
    }
};

class EnumFieldConversionPattern : public OpConversionPattern<P4HIR::ConstOp> {
 public:
    using OpConversionPattern<P4HIR::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const override {
        auto oldAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(op.getValue());
        if (!oldAttr) return mlir::failure();

        auto originalType = mlir::dyn_cast<P4HIR::EnumType>(oldAttr.getType());
        if (!originalType) return mlir::failure();

        auto targetType = getTypeConverter()->convertType(originalType);
        if (!targetType) return mlir::failure();

        auto newAttr = P4HIR::EnumFieldAttr::get(targetType, oldAttr.getField());
        rewriter.replaceOpWithNewOp<P4HIR::ConstOp>(op, newAttr);

        return mlir::success();
    }
};

void EnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    EnumTypeConverter typeConverter(&context);
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<P4HIR::FuncOp>([&](P4HIR::FuncOp func) {
        auto fnType = func.getFunctionType();
        return typeConverter.isLegal(fnType.getInputs()) &&
               typeConverter.isLegal(fnType.getReturnTypes());
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
    });

    target.addLegalOp<P4HIR::CaseOp>();

    RewritePatternSet patterns(&context);
    patterns.add<EnumFieldConversionPattern>(typeConverter, &context);
    P4::P4MLIR::populateFunctionOpInterfaceTypeConversionPattern<P4HIR::FuncOp>(patterns,
                                                                                typeConverter);
    populateGenericOpTypeConversionPattern<P4HIR::CallOp, P4HIR::InstantiateOp, P4HIR::ApplyOp,
                                           P4HIR::VariableOp, P4HIR::AssignOp, P4HIR::ReadOp,
                                           P4HIR::CmpOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createEnumEliminationPass() {
    return std::make_unique<EnumEliminationPass>();
}
