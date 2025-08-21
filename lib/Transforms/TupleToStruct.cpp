// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "p4mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"


#define DEBUG_TYPE "convert-p4hir-tuple-to-struct"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_TUPLETOSTRUCT
#include "p4mlir/Transforms/Passes.cpp.inc"
}

using namespace P4::P4MLIR;

namespace {

struct TupleToStructPass
    : public P4::P4MLIR::impl::TupleToStructBase<TupleToStructPass> {
    TupleToStructPass() = default;
    void runOnOperation() override;
};


class TupleToStructTypeConverter: public P4HIRTypeConverter {
 public:
    explicit TupleToStructTypeConverter() {

        addConversion([](TupleType tupleType)-> Type {
            MLIRContext *ctx = tupleType.getContext();
            ArrayRef<Type> elementTypes = tupleType.getTypes();

            SmallVector<P4::P4MLIR::P4HIR::FieldInfo> structFiedls;
            structFiedls.reserve(elementTypes.size());

            for(unsigned i=0; i< elementTypes.size(); i++){
                mlir::StringAttr name = mlir::StringAttr::get(ctx, "elemet_" + std::to_string(i));
                P4::P4MLIR::P4HIR::FieldInfo field = P4::P4MLIR::P4HIR::FieldInfo(name, elementTypes[i]);
                structFiedls.push_back(field);
            }
            llvm::StringRef name("_tupletoStruct");
            return P4::P4MLIR::P4HIR::StructType::get(ctx, name, structFiedls);
        });
    }
};

struct TupleOpConversion : public OpConversionPattern<P4::P4MLIR::P4HIR::TupleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(P4::P4MLIR::P4HIR::TupleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structType = getTypeConverter()->convertType(op.getType()).dyn_cast<P4::P4MLIR::P4HIR::StructType>();

    if(!structType)
      return rewriter.notifyMatchFailure(op, "Type conversion failed to struct");

    Value newStruct = rewriter.create<P4::P4MLIR::P4HIR::StructOp>(loc, structType, adaptor.getOperands());
    rewriter.replaceOp(op, newStruct);
    return success();

   }
};

struct TupleExtractOpConversion : public OpConversionPattern<P4::P4MLIR::P4HIR::TupleExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(P4::P4MLIR::P4HIR::TupleExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    auto index = op.getFieldIndex();
    Value input =  adaptor.getOperands()[0];
    auto structType = input.getType().dyn_cast<P4::P4MLIR::P4HIR::StructType>();

    if(!structType)
      return rewriter.notifyMatchFailure(op, "Expect StructType as Input");

    if(index >= structType.getFields().size())
      return rewriter.notifyMatchFailure(op, "Indexout of bounds in StructType");

    const auto &field = structType.getFields()[index];
    StringAttr fieldName = field.name;

    rewriter.replaceOpWithNewOp<P4::P4MLIR::P4HIR::StructExtractOp>(op, input, fieldName);
    return success();

   }
};
}  // namescpace

void TupleToStructPass::runOnOperation() {
        mlir::ModuleOp module = getOperation();
        MLIRContext &context  = getContext();

        P4HIRTypeConverter typeConverter;

        RewritePatternSet patterns(&context);
        ConversionTarget target(context);

        target.addIllegalOp<P4::P4MLIR::P4HIR::TupleOp>();
        target.addIllegalOp<P4::P4MLIR::P4HIR::TupleExtractOp>();

        target.markUnknownOpDynamicallyLegal([&](Operation *op){
            return typeConverter.isLegal(op);
        });

        patterns.add<TupleOpConversion, TupleExtractOpConversion>(typeConverter, &context);

        if(failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createTupleToStructPass() {
    return std::make_unique<TupleToStructPass>();
}