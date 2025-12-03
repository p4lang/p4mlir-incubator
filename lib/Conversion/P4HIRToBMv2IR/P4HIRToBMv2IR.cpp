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

namespace P4::P4MLIR {
#define GEN_PASS_DEF_P4HIRTOBMV2IR
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

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
            return BMv2IR::HeaderType::get(headerType.getContext(), headerType.getName(),
                                           newFields);
        });
    }
};

struct ExtractOpConversionPattern : public OpConversionPattern<P4CoreLib::PacketExtractOp> {
    using OpConversionPattern<P4CoreLib::PacketExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4CoreLib::PacketExtractOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto context = op.getContext();
        auto hdr = op.getHdr();
        auto referredTy = hdr.getType().getObjectType();
        if (!isa<P4HIR::HeaderType>(referredTy))
            return op->emitError("Only headers supported as BMv2 extract arguments");
        auto fieldRefOp = op.getHdr().getDefiningOp<P4HIR::StructFieldRefOp>();
        if (!fieldRefOp) return op->emitError("Unsupported extract argument");
        auto fieldName = fieldRefOp.getFieldName();
        // TODO: support non-regular extracts
        rewriter.replaceOpWithNewOp<BMv2IR::ExtractOp>(
            op, BMv2IR::ExtractKindAttr::get(context, BMv2IR::ExtractKind::Regular),
            rewriter.getStringAttr(fieldName), nullptr);
        rewriter.eraseOp(fieldRefOp);
        return success();
    }
};

struct ParserStateOpConversionPattern : public OpConversionPattern<P4HIR::ParserStateOp> {
    using OpConversionPattern<P4HIR::ParserStateOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(P4HIR::ParserStateOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();

        SmallVector<Operation *> eraseList;

        auto newState = rewriter.create<BMv2IR::ParserStateOp>(loc, op.getSymNameAttr());
        auto &transitionBlock = newState.getTransitions().emplaceBlock();
        auto &keysBlock = newState.getTransitionKeys().emplaceBlock();
        auto terminator = op.getNextTransition();

        auto conversionOk =
            TypeSwitch<Operation *, LogicalResult>(terminator)
                .Case([&](P4HIR::ParserTransitionOp transitionOp) -> LogicalResult {
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        transitionOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        transitionOp.getStateAttr(), nullptr, nullptr);
                    eraseList.push_back(transitionOp.getOperation());
                    return success();
                })
                .Case([&](P4HIR::ParserTransitionSelectOp transitionSelectOp) {
                    for (auto operand : transitionSelectOp.getArgs()) {
                        auto transitionKey = insertTransitionKey(operand.getDefiningOp(), rewriter,
                                                                 &keysBlock, eraseList);
                        if (failed(transitionKey)) return failure();
                    }

                    for (auto selectOp : transitionSelectOp.selects()) {
                        auto transition = insertTransition(selectOp, rewriter, &transitionBlock);
                        if (failed(transition)) return failure();
                    }
                    eraseList.push_back(transitionSelectOp);
                    return success();
                })
                .Case([&](P4HIR::ParserAcceptOp acceptOp) {
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        acceptOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        nullptr, nullptr, nullptr);
                    eraseList.push_back(acceptOp.getOperation());
                    return success();
                })
                .Case([&](P4HIR::ParserRejectOp rejectOp) {
                    // TODO: p4c raises a warning "Explicit transition to reject not supported on
                    // this target"
                    //  for explicit transitions to reject
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        rejectOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        nullptr, nullptr, nullptr);
                    eraseList.push_back(rejectOp.getOperation());
                    return success();
                })
                .Default([](Operation *) { return failure(); });

        if (failed(conversionOk)) return op->emitError("Error while processing state terminator");

        // Move the remaning parser ops to their region, they will be converted by the other
        // patterns
        newState.getParserOps().takeBody(op.getBody());
        for (auto op : eraseList) rewriter.eraseOp(op);
        rewriter.replaceOp(op, newState);

        return success();
    }

 private:
    static LogicalResult insertTransitionKey(Operation *op, ConversionPatternRewriter &rewriter,
                                             Block *block, SmallVector<Operation *> &eraseList) {
        auto loc = op->getLoc();
        if (auto lookAheadOp = dyn_cast<P4CoreLib::PacketLookAheadOp>(op)) {
            // TODO: not sure how to handle offsets
            auto offset = rewriter.getI32IntegerAttr(0);
            // TODO: can PacketLookAheadOp return something other than Bit?
            auto bitTy = cast<P4HIR::BitsType>(lookAheadOp.getResult().getType());
            auto width = rewriter.getI32IntegerAttr(bitTy.getWidth());
            ConversionPatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(block);
            eraseList.push_back(lookAheadOp);
            rewriter.create<BMv2IR::LookaheadOp>(loc, offset, width);
            return success();
        }
        return op->emitError("Unhandled transition key");
    }

    static LogicalResult insertTransition(P4HIR::ParserSelectCaseOp caseOp,
                                          ConversionPatternRewriter &rewriter, Block *block) {
        auto context = caseOp.getContext();
        auto keysets = caseOp.getSelectKeys();
        auto loc = caseOp.getLoc();

        auto constValOrError = [](Value val) -> llvm::FailureOr<TypedAttr> {
            auto constOp = val.getDefiningOp<P4HIR::ConstOp>();
            if (!constOp) return failure();

            return constOp.getValue();
        };

        for (auto entry : keysets) {
            return TypeSwitch<Operation *, LogicalResult>(entry.getDefiningOp())
                .Case([&](P4HIR::SetOp setOp) -> LogicalResult {
                    auto inputs = setOp.getInput();
                    for (auto input : inputs) {
                        auto maybeConstVal = constValOrError(input);
                        if (failed(maybeConstVal)) return setOp.emitError("Unsupported set entry");
                        ConversionPatternRewriter::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToEnd(block);

                        rewriter.create<BMv2IR::TransitionOp>(
                            loc,
                            BMv2IR::TransitionKindAttr::get(context,
                                                            BMv2IR::TransitionKind::Hexstr),
                            caseOp.getStateAttr(), maybeConstVal.value(), nullptr);
                    }
                    return success();
                })
                .Case([&](P4HIR::ConstOp constOp) {
                    if (P4HIR::isUniversalSetValue(constOp.getRes())) {
                        ConversionPatternRewriter::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToEnd(block);
                        rewriter.create<BMv2IR::TransitionOp>(
                            loc,
                            BMv2IR::TransitionKindAttr::get(context,
                                                            BMv2IR::TransitionKind::Default),
                            caseOp.getStateAttr(), nullptr, nullptr);
                        return success();
                    }
                    return failure();
                })
                .Case([&](P4HIR::MaskOp maskOp) -> LogicalResult {
                    auto maybeLhsConst = constValOrError(maskOp.getLhs());
                    auto maybeRhsConst = constValOrError(maskOp.getRhs());
                    if (failed(maybeLhsConst) || failed(maybeRhsConst))
                        return maskOp.emitError("Unhandled mask op");
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(block);
                    rewriter.create<BMv2IR::TransitionOp>(
                        maskOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Hexstr),
                        caseOp.getStateAttr(), maybeLhsConst.value(), maybeRhsConst.value());
                    return success();
                })
                .Default([&](Operation *) -> LogicalResult {
                    return caseOp.emitError("Unsupported select case");
                });
        }
        return failure();
    }
};

struct ParserOpConversionPattern : public OpConversionPattern<P4HIR::ParserOp> {
    using OpConversionPattern<P4HIR::ParserOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ParserOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto firstTransition = op.getStartTransition();
        auto initState = op.getStartState();
        auto newParser =
            rewriter.create<BMv2IR::ParserOp>(loc, op.getSymNameAttr(), initState.getSymbolRef());
        rewriter.eraseOp(firstTransition);
        auto &region = newParser.getRegion();
        region.takeBody(op.getRegion());
        rewriter.replaceOp(op, newParser);
        return success();
    }
};

struct P4HIRToBMv2IRPass : public P4::P4MLIR::impl::P4HIRToBmv2IRBase<P4HIRToBMv2IRPass> {
    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp module = getOperation();
        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        P4HIRToBMv2IRTypeConverter converter;
        patterns.add<ParserOpConversionPattern, ParserStateOpConversionPattern,
                     ExtractOpConversionPattern>(converter, &context);

        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<P4HIR::ParserOp>();
        target.addIllegalOp<P4HIR::ParserStateOp>();
        target.addIllegalOp<P4CoreLib::PacketExtractOp>();
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
}  // anonymous namespace
