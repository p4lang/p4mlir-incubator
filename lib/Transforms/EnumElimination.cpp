#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
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
        // Converts EnumType to SerEnumType
        addConversion([ctx](P4HIR::EnumType enumType) -> Type {
            // TODO: Use external model/policy to define underlying type instead of always bit<32>
            auto underlyingType = P4HIR::BitsType::get(ctx, 32, false);

            // Create field mappings (for bit<32> is index but this should be externally defined)
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

        // Fallback for other types.
        addConversion([](Type type) -> std::optional<Type> {
            if (isa<P4HIR::EnumType>(type)) {
                return std::nullopt;
            }
            return type;
        });
    }
};

// Converts enum field references to use the new SerEnumType
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

class CallOpConversionPattern : public OpConversionPattern<P4HIR::CallOp> {
 public:
    using OpConversionPattern<P4HIR::CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::CallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> typeOperands;
        if (auto typeOperandsAttr = op.getTypeOperandsAttr()) {
            for (Attribute typeAttr : typeOperandsAttr.getValue()) {
                typeOperands.push_back(cast<TypeAttr>(typeAttr).getValue());
            }
        }

        P4HIR::CallOp newCall;
        if (op.getResult()) {
            Type resultType = getTypeConverter()->convertType(op.getResult().getType());
            if (!resultType) {
                return mlir::failure();
            }
            newCall = rewriter.create<P4HIR::CallOp>(op.getLoc(), op.getCalleeAttr(), resultType,
                                                     typeOperands, adaptor.getOperands());
        } else {
            newCall = rewriter.create<P4HIR::CallOp>(op.getLoc(), op.getCalleeAttr(), typeOperands,
                                                     adaptor.getOperands());
        }

        rewriter.replaceOp(op, newCall.getResults());
        return mlir::success();
    }
};

// TODO: Implement FuncOpConversionPattern

void EnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    EnumTypeConverter converter(&context);
    ConversionTarget target(context);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return converter.isLegal(op->getOperandTypes()) && converter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(&context);
    patterns.add<EnumFieldConversionPattern, CallOpConversionPattern>(converter, &context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> P4::P4MLIR::createEnumEliminationPass() {
    return std::make_unique<EnumEliminationPass>();
}
