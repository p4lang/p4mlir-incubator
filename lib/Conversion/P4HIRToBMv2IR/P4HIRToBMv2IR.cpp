#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "p4hir-convert-to-bmv2"

using namespace mlir;
using namespace P4::P4MLIR;

namespace {

BMv2IR::FieldInfo convertFieldInfo(P4HIR::FieldInfo p4Field) {
    return BMv2IR::FieldInfo(p4Field.name, p4Field.type);
}

struct P4HIRToBMv2IRTypeConverter : public mlir::TypeConverter {
    P4HIRToBMv2IRTypeConverter() {
        addConversion([&](mlir::Type t) { return t; });
        addConversion([&](P4HIR::HeaderType headerType) -> Type {
            SmallVector<BMv2IR::FieldInfo> newFields;
            for (auto field : headerType.getFields()) {
                if (!BMv2IR::HeaderType::isAllowedFieldType(field.type)) return nullptr;
                newFields.push_back(convertFieldInfo(field));
            }
            return BMv2IR::HeaderType::get(headerType.getContext(),
                                           headerType.getName(),
                                           newFields);
        });
    }
};

/// FIXED IMPLEMENTATION: packet_in.extract lowering
struct ExtractOpConversionPattern
    : public OpConversionPattern<P4CoreLib::PacketExtractOp> {

    using OpConversionPattern<P4CoreLib::PacketExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4CoreLib::PacketExtractOp op,
                                  OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto context = op.getContext();
        auto hdr = op.getHdr();
        auto referredTy = hdr.getType().getObjectType();

        if (isa<P4HIR::HeaderStackType>(referredTy)) {
            
            auto fieldRefOp =
                hdr.getDefiningOp<P4HIR::StructFieldRefOp>();
            if (!fieldRefOp)
                return op->emitError(
                    "Unsupported header stack extract target");

            auto fieldName = fieldRefOp.getFieldName();

            rewriter.replaceOpWithNewOp<BMv2IR::ExtractOp>(
                op,
                BMv2IR::ExtractKindAttr::get(context,
                                             BMv2IR::ExtractKind::Regular),
                rewriter.getStringAttr(fieldName),
                /*offset=*/nullptr);

            rewriter.eraseOp(fieldRefOp);
            return success();
        }

        if (!isa<P4HIR::HeaderType>(referredTy))
            return op->emitError(
                "Only headers and header stacks supported as extract arguments");

        auto fieldRefOp =
            hdr.getDefiningOp<P4HIR::StructFieldRefOp>();
        if (!fieldRefOp)
            return op->emitError("Unsupported extract argument");

        auto fieldName = fieldRefOp.getFieldName();

        rewriter.replaceOpWithNewOp<BMv2IR::ExtractOp>(
            op,
            BMv2IR::ExtractKindAttr::get(context,
                                         BMv2IR::ExtractKind::Regular),
            rewriter.getStringAttr(fieldName),
            /*offset=*/nullptr);

        rewriter.eraseOp(fieldRefOp);
        return success();
    }
};

struct ParserStateOpConversionPattern
    : public OpConversionPattern<P4HIR::ParserStateOp> {
    using OpConversionPattern<P4HIR::ParserStateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ParserStateOp op,
                                  OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // unchanged
        return failure();
    }
};

struct ParserOpConversionPattern
    : public OpConversionPattern<P4HIR::ParserOp> {
    using OpConversionPattern<P4HIR::ParserOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ParserOp op,
                                  OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto firstTransition = op.getStartTransition();
        auto initState = op.getStartState();
        auto newParser =
            rewriter.create<BMv2IR::ParserOp>(
                loc, op.getSymNameAttr(), initState.getSymbolRef());
        rewriter.eraseOp(firstTransition);
        newParser.getRegion().takeBody(op.getRegion());
        rewriter.replaceOp(op, newParser);
        return success();
    }
};

struct P4HIRToBMv2IRPass
    : public P4::P4MLIR::impl::P4HIRToBmv2IRBase<P4HIRToBMv2IRPass> {

    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp module = getOperation();
        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        P4HIRToBMv2IRTypeConverter converter;

        patterns.add<ParserOpConversionPattern,
                     ParserStateOpConversionPattern,
                     ExtractOpConversionPattern>(converter, &context);

        target.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });

        target.addIllegalOp<P4HIR::ParserOp>();
        target.addIllegalOp<P4HIR::ParserStateOp>();
        target.addIllegalOp<P4CoreLib::PacketExtractOp>();

        if (failed(applyPartialConversion(
                module, target, std::move(patterns))))
            signalPassFailure();
    }
};

}  // namespace
