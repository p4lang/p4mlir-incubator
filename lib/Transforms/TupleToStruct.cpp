// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-tuple-to-struct"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_TUPLETOSTRUCT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct TupleToStructPass : public P4::P4MLIR::impl::TupleToStructBase<TupleToStructPass> {
    TupleToStructPass() = default;
    void runOnOperation() override;
};

class TupleToStructTypeConverter : public P4HIRTypeConverter {
 public:
    explicit TupleToStructTypeConverter() {
        addConversion([this](TupleType tupleType) -> Type {
            MLIRContext *ctx = tupleType.getContext();
            SmallVector<Type> convertedElements;

            for (Type t : tupleType.getTypes()) {
                auto ce = this->convertType(t);
                if (!ce) return nullptr;
                convertedElements.emplace_back(ce);
            }

            SmallVector<P4HIR::FieldInfo> structFields;
            mlir::StringAttr name;

            for (auto [index, type] : llvm::enumerate(convertedElements)) {
                name = mlir::StringAttr::get(ctx, "element_" + std::to_string(index));
                structFields.emplace_back(name, type);
            }
            return P4HIR::StructType::get(ctx, "_tupleToStruct", structFields);
        });
    }
};

struct TupleOpConversion : public OpConversionPattern<P4HIR::TupleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::TupleOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto structType =
            llvm::dyn_cast<P4HIR::StructType>(getTypeConverter()->convertType(op.getType()));

        if (!structType)
            return rewriter.notifyMatchFailure(op, "Type conversion failed to P4HIR StructType");

        rewriter.replaceOpWithNewOp<P4HIR::StructOp>(op, structType, adaptor.getOperands());
        return success();
    }
};

struct TupleExtractOpConversion : public OpConversionPattern<P4HIR::TupleExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::TupleExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto index = op.getFieldIndex();
        Value input = adaptor.getOperands()[0];
        auto structType = llvm::dyn_cast<P4HIR::StructType>(input.getType());

        if (!structType)
            return rewriter.notifyMatchFailure(op, "expected P4HIR StructType as input operand");

        if (index >= structType.getFields().size())
            return rewriter.notifyMatchFailure(op, "index out of bounds in P4HIR StructType");

        rewriter.replaceOpWithNewOp<P4HIR::StructExtractOp>(op, input,
                                                            structType.getFields()[index].name);
        return success();
    }
};

}  // end namespace

void TupleToStructPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    TupleToStructTypeConverter typeConverter;

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    populateTypeConversionPattern(patterns, typeConverter);

    patterns.add<TupleOpConversion, TupleExtractOpConversion>(typeConverter, &context);

    target.addIllegalOp<P4HIR::TupleOp, P4HIR::TupleExtractOp>();

    configureUnknownOpDynamicallyLegalByTypes(target, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> createTupleToStructPass() { return std::make_unique<TupleToStructPass>(); }

}  // namespace P4::P4MLIR