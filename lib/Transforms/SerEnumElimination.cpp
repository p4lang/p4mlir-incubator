#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/DialectConversion.h"
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

class SerEnumTypeConverter : public TypeConverter {
 public:
    explicit SerEnumTypeConverter(MLIRContext *ctx) {
        // Fallback for other types
        addConversion([](Type type) -> Type { return type; });

        // Converts SerEnumType to the underlying BitsType
        addConversion([](P4HIR::SerEnumType serEnumType) -> Type { return serEnumType.getType(); });

        // Convert reference types by converting their object type
        addConversion([this](P4HIR::ReferenceType refType) -> std::optional<Type> {
            auto newType = convertType(refType.getObjectType());
            if (!newType) return std::nullopt;

            return P4HIR::ReferenceType::get(newType);
        });
    }
};

class SerEnumFieldConversionPattern : public OpConversionPattern<P4HIR::ConstOp> {
 public:
    using OpConversionPattern<P4HIR::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const override {
        auto enumFieldAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(op.getValue());
        if (!enumFieldAttr) return mlir::failure();

        auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(enumFieldAttr.getType());
        if (!serEnumType) return mlir::failure();

        P4HIR::IntAttr fieldInt = mlir::cast_if_present<P4HIR::IntAttr>(
            serEnumType.getFields().get(enumFieldAttr.getField()));

        rewriter.replaceOpWithNewOp<P4HIR::ConstOp>(op, fieldInt);

        return success();
    }
};

class SwitchConversionPattern : public OpConversionPattern<P4HIR::SwitchOp> {
 public:
    using OpConversionPattern<P4HIR::SwitchOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::SwitchOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        TypeConverter typeConverter = *getTypeConverter();

        TypeConverter::SignatureConversion result(1);
        if (failed(typeConverter.convertSignatureArgs(TypeRange(op.getCondition()), result)) ||
            failed(rewriter.convertRegionTypes(&op.getBody(), typeConverter, &result))) {
            return mlir::failure();
        }

        rewriter.modifyOpInPlace(
            op, [&]() { op.getConditionMutable().assign(adaptor.getCondition()); });

        return mlir::success();
    }
};

class CaseConversionPattern : public OpConversionPattern<P4HIR::CaseOp> {
 public:
    using OpConversionPattern<P4HIR::CaseOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::CaseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        TypeConverter typeConverter = *getTypeConverter();

        TypeConverter::SignatureConversion result(op.getValue().size());
        if (failed(rewriter.convertRegionTypes(&op.getCaseRegion(), typeConverter, &result))) {
            return mlir::failure();
        }

        SmallVector<Attribute> newValues;
        for (Attribute val : op.getValue()) {
            auto enumFieldAttr = dyn_cast<P4HIR::EnumFieldAttr>(val);
            if (!enumFieldAttr) return mlir::failure();

            auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(enumFieldAttr.getType());
            if (!serEnumType) return mlir::failure();

            P4HIR::IntAttr fieldInt =
                mlir::cast<P4HIR::IntAttr>(serEnumType.getFields().get(enumFieldAttr.getField()));
            newValues.push_back(fieldInt);
        }

        rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(rewriter.getArrayAttr(newValues)); });

        return mlir::success();
    }
};

}  // namespace

void SerEnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    SerEnumTypeConverter typeConverter(&context);
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<P4HIR::FuncOp>([&](P4HIR::FuncOp func) {
        auto fnType = func.getFunctionType();
        return typeConverter.isLegal(fnType.getInputs()) &&
               typeConverter.isLegal(fnType.getReturnTypes());
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
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(&context);
    patterns.add<SerEnumFieldConversionPattern, SwitchConversionPattern, CaseConversionPattern>(
        typeConverter, &context);
    P4::P4MLIR::populateFunctionOpInterfaceTypeConversionPattern<P4HIR::FuncOp>(patterns,
                                                                                typeConverter);
    populateGenericOpTypeConversionPattern<P4HIR::CallOp, P4HIR::InstantiateOp, P4HIR::ApplyOp,
                                           P4HIR::VariableOp, P4HIR::AssignOp, P4HIR::ReadOp,
                                           P4HIR::CmpOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createSerEnumEliminationPass() {
    return std::make_unique<SerEnumEliminationPass>();
}
