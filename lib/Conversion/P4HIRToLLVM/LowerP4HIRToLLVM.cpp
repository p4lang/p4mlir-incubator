// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/P4HIRToLLVM/P4HIRToLLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"  // IWYU pragma: keep (required for Passes.cpp.inc)
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define DEBUG_TYPE "p4hir-to-llvm"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/P4HIRToLLVM/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

struct ConstOpConversion : public ConvertOpToLLVMPattern<P4HIR::ConstOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto value = adaptor.getValue();
        // Passing the interface handle by its base type drops a cached concept pointer, which
        // is exactly what `convertTypeAttribute` takes.
        // NOLINTNEXTLINE(cppcoreguidelines-slicing)
        auto newAttr = getTypeConverter()->convertTypeAttribute(value.getType(), value);
        if (!newAttr) {
            return rewriter.notifyMatchFailure(op, "unsupported constant type");
        }
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, cast<TypedAttr>(*newAttr));
        return success();
    }
};

template <typename Op>
LogicalResult lowerToOp(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<Op>(op, operands);
    return success();
}

template <typename Op>
LogicalResult lowerToOp(Operation *op, P4HIR::BinOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
    return lowerToOp<Op>(op, adaptor.getOperands(), rewriter);
}

// LLVM integers are signless; signedness comes from the original BitsType.
template <typename SignedOp, typename UnsignedOp>
LogicalResult lowerToSignedOrUnsignedOp(P4HIR::BinOp op, P4HIR::BinOp::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
    if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(op.getType())) {
        if (bitsType.isSigned()) {
            rewriter.replaceOpWithNewOp<SignedOp>(op, adaptor.getOperands());
        } else {
            rewriter.replaceOpWithNewOp<UnsignedOp>(op, adaptor.getOperands());
        }
        return success();
    }
    return rewriter.notifyMatchFailure(op, "expected bits type");
}

template <typename UnsignedOp>
LogicalResult lowerToUnsignedDivisionOp(P4HIR::BinOp op, P4HIR::BinOp::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
    auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(op.getType());
    if (!bitsType) {
        return rewriter.notifyMatchFailure(op, "expected bits type");
    }
    if (bitsType.isSigned()) {
        return rewriter.notifyMatchFailure(op, "not defined on signed values");
    }
    return lowerToOp<UnsignedOp>(op, adaptor, rewriter);
}

struct BinOpConversion : public ConvertOpToLLVMPattern<P4HIR::BinOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::BinOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        switch (op.getKind()) {
            case P4HIR::BinOpKind::Add:
                return lowerToOp<LLVM::AddOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::AddSat:
                return lowerToSignedOrUnsignedOp<LLVM::SAddSat, LLVM::UAddSat>(op, adaptor,
                                                                               rewriter);
            case P4HIR::BinOpKind::Sub:
                return lowerToOp<LLVM::SubOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::SubSat:
                return lowerToSignedOrUnsignedOp<LLVM::SSubSat, LLVM::USubSat>(op, adaptor,
                                                                               rewriter);
            case P4HIR::BinOpKind::Mul:
                return lowerToOp<LLVM::MulOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Div:
                return lowerToUnsignedDivisionOp<LLVM::UDivOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Mod:
                return lowerToUnsignedDivisionOp<LLVM::URemOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::And:
                return lowerToOp<LLVM::AndOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Or:
                return lowerToOp<LLVM::OrOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Xor:
                return lowerToOp<LLVM::XOrOp>(op, adaptor, rewriter);
        }
        return rewriter.notifyMatchFailure(op, "unsupported binop kind");
    }
};

struct UnaryOpConversion : public ConvertOpToLLVMPattern<P4HIR::UnaryOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::UnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto input = adaptor.getInput();
        auto intType = cast<IntegerType>(input.getType());
        auto width = intType.getWidth();

        auto createConstant = [&](const APInt &value) -> Value {
            return LLVM::ConstantOp::create(rewriter, op.getLoc(), intType, value);
        };

        switch (op.getKind()) {
            case P4HIR::UnaryOpKind::UPlus:
                rewriter.replaceOp(op, input);
                return success();
            case P4HIR::UnaryOpKind::Neg:  // `-x` is emitted as `0 - x`
                return lowerToOp<LLVM::SubOp>(op, {createConstant(APInt::getZero(width)), input},
                                              rewriter);
            case P4HIR::UnaryOpKind::Cmpl:  // `~x` is emitted as `x ^ all-ones`
                return lowerToOp<LLVM::XOrOp>(op, {input, createConstant(APInt::getAllOnes(width))},
                                              rewriter);
            case P4HIR::UnaryOpKind::LNot:  // `!x` is emitted as `x ^ 1`
                return lowerToOp<LLVM::XOrOp>(op, {input, createConstant(APInt(width, 1))},
                                              rewriter);
        }
        return rewriter.notifyMatchFailure(op, "unsupported unary op kind");
    }
};

struct CmpOpConversion : public ConvertOpToLLVMPattern<P4HIR::CmpOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto lhsType = op.getLhs().getType();
        if (!isa<P4HIR::BitsType, P4HIR::BoolType>(lhsType)) {
            return rewriter.notifyMatchFailure(op, "unsupported cmp operand type");
        }

        bool isSigned = false;  // BoolType lowers to i1 and is always compared as unsigned.
        if (auto bitsType = dyn_cast<P4HIR::BitsType>(lhsType)) isSigned = bitsType.isSigned();

        auto lowerToICmpOp = [&](LLVM::ICmpPredicate predicate) {
            rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, predicate, adaptor.getLhs(),
                                                      adaptor.getRhs());
            return success();
        };

        switch (op.getKind()) {
            case P4HIR::CmpOpKind::Eq:
                return lowerToICmpOp(LLVM::ICmpPredicate::eq);
            case P4HIR::CmpOpKind::Ne:
                return lowerToICmpOp(LLVM::ICmpPredicate::ne);
            case P4HIR::CmpOpKind::Lt:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::slt
                                              : LLVM::ICmpPredicate::ult);
            case P4HIR::CmpOpKind::Le:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sle
                                              : LLVM::ICmpPredicate::ule);
            case P4HIR::CmpOpKind::Gt:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sgt
                                              : LLVM::ICmpPredicate::ugt);
            case P4HIR::CmpOpKind::Ge:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sge
                                              : LLVM::ICmpPredicate::uge);
        }
        return rewriter.notifyMatchFailure(op, "unsupported cmp op kind");
    }
};

Value createZExtOrTrunc(Value value, IntegerType resultType, Location loc,
                        ConversionPatternRewriter &rewriter) {
    auto valueType = cast<IntegerType>(value.getType());
    if (valueType.getWidth() < resultType.getWidth())
        return LLVM::ZExtOp::create(rewriter, loc, resultType, value);
    if (valueType.getWidth() > resultType.getWidth())
        return LLVM::TruncOp::create(rewriter, loc, resultType, value);
    return value;
}

// LLVM shifts are poison past the bit width; P4 shifts are safe for those cases.
template <typename LLVMShiftOp>
LogicalResult lowerToShiftOp(Operation *op, Value lhs, Value rhs,
                             ConversionPatternRewriter &rewriter) {
    auto resultType = dyn_cast<IntegerType>(lhs.getType());
    auto shiftType = dyn_cast<IntegerType>(rhs.getType());
    if (!resultType || !shiftType)
        return rewriter.notifyMatchFailure(op, "expected converted integer operands");

    auto loc = op->getLoc();
    auto resultWidth = resultType.getWidth();
    auto shift = createZExtOrTrunc(rhs, resultType, loc, rewriter);
    // Truncation drops the very bits that put a shift out of range, so the
    // overflow check has to see the operand at its original width.
    auto untruncatedShift = shiftType.getWidth() >= resultWidth ? rhs : shift;
    Value widthValue = LLVM::ConstantOp::create(
        rewriter, loc, cast<IntegerType>(untruncatedShift.getType()), resultWidth);
    Value overflow =
        LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, untruncatedShift, widthValue);

    if constexpr (std::is_same_v<LLVMShiftOp, LLVM::AShrOp>) {
        // P4 defines an out-of-range right shift as all sign bits, which `ashr`
        // by width - 1 already produces: clamping the shift is the whole fix,
        // and unlike the logical shifts below the result needs no second select.
        Value maxShift = LLVM::ConstantOp::create(rewriter, loc, resultType, resultWidth - 1);
        Value safeShift = LLVM::SelectOp::create(rewriter, loc, overflow, maxShift, shift);
        rewriter.replaceOpWithNewOp<LLVMShiftOp>(op, lhs, safeShift);
    } else {
        Value zero = LLVM::ConstantOp::create(rewriter, loc, resultType, 0);
        Value safeShift = LLVM::SelectOp::create(rewriter, loc, overflow, zero, shift);
        Value inRangeResult = LLVMShiftOp::create(rewriter, loc, lhs, safeShift);
        rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, overflow, zero, inRangeResult);
    }
    return success();
}

struct ShlOpConversion : public ConvertOpToLLVMPattern<P4HIR::ShlOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ShlOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!isa<P4HIR::BitsType>(op.getLhs().getType()) ||
            !isa<P4HIR::BitsType>(op.getRhs().getType()))
            return rewriter.notifyMatchFailure(op, "expected fixed-width bits operands");
        return lowerToShiftOp<LLVM::ShlOp>(op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    }
};

struct ShrOpConversion : public ConvertOpToLLVMPattern<P4HIR::ShrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ShrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto lhsBitsType = dyn_cast<P4HIR::BitsType>(op.getLhs().getType());
				auto rhsIsBitsType = isa<P4HIR::BitsType>(op.getRhs().getType());
        if (!lhsBitsType || !rhsIsBitsType)
            return rewriter.notifyMatchFailure(op, "expected fixed-width bits operands");
        if (lhsBitsType.isSigned())
            return lowerToShiftOp<LLVM::AShrOp>(op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
        return lowerToShiftOp<LLVM::LShrOp>(op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    }
};

struct ConcatOpConversion : public ConvertOpToLLVMPattern<P4HIR::ConcatOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ConcatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto highType = cast<IntegerType>(adaptor.getLhs().getType());
        auto lowType = cast<IntegerType>(adaptor.getRhs().getType());

        auto loc = op.getLoc();
        auto lowWidth = lowType.getWidth();
        auto resultType = rewriter.getIntegerType(highType.getWidth() + lowWidth);

        // Concatenation operates on the bit patterns of its operands, regardless of
        // their signedness. Zero-extend both operands so that sign extension cannot
        // introduce bits into the other half of the result.
        Value high = createZExtOrTrunc(adaptor.getLhs(), resultType, loc, rewriter);
        Value low = createZExtOrTrunc(adaptor.getRhs(), resultType, loc, rewriter);
        // The shift amount is the width of the low half, which is always smaller
        // than the result width.
        Value shift = LLVM::ConstantOp::create(rewriter, loc, resultType, lowWidth);
        high = LLVM::ShlOp::create(rewriter, loc, high, shift);
        rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, high, low);
        return success();
    }
};

struct SliceOpConversion : public ConvertOpToLLVMPattern<P4HIR::SliceOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::SliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto inputType = cast<IntegerType>(adaptor.getInput().getType());

        auto loc = op.getLoc();
        auto resultType = rewriter.getIntegerType(op.getHighBit() - op.getLowBit() + 1);

        Value value = adaptor.getInput();
        if (uint32_t lowBit = op.getLowBit(); lowBit > 0) {
            // Slicing operates on the bit pattern, so shift logically regardless of the
            // signedness of the input.
            Value shift = LLVM::ConstantOp::create(rewriter, loc, inputType, lowBit);
            value = LLVM::LShrOp::create(rewriter, loc, value, shift);
        }
        rewriter.replaceOp(op, createZExtOrTrunc(value, resultType, loc, rewriter));
        return success();
    }
};

struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    void runOnOperation() override {
        auto &context = getContext();
        auto module = getOperation();

        LLVMTypeConverter typeConverter(&context);
        populateP4HIRToLLVMTypeConversion(typeConverter);

        LLVMConversionTarget target(context);
        target.addLegalOp<ModuleOp>();

        RewritePatternSet patterns(&context);
        populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);

        // Lowering should be driven by the patterns above, not by constant
        // folding P4HIR ops (e.g. `p4hir.binop(add, ...)` on two constants)
        // before they even get a chance to match.
        ConversionConfig config;
        config.foldingMode = DialectConversionFoldingMode::Never;

        if (failed(applyPartialConversion(module, target, std::move(patterns), config))) {
            signalPassFailure();
        }
    }
};

}  // namespace

void P4::P4MLIR::populateP4HIRToLLVMTypeConversion(LLVMTypeConverter &converter) {
    converter.addConversion([](P4HIR::BitsType bitsType) -> std::optional<Type> {
        // P4 allows `bit<0>`, LLVM has no `i0`: leave such values unconverted.
        if (bitsType.getWidth() == 0) return std::nullopt;
        return IntegerType::get(bitsType.getContext(), bitsType.getWidth());
    });

    converter.addConversion(
        [](P4HIR::BoolType boolType) { return IntegerType::get(boolType.getContext(), 1); });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BitsType bitsType,
                     P4HIR::IntAttr attr) -> LLVMTypeConverter::AttributeConversionResult {
            // Types without an LLVM counterpart (e.g. `bit<0>`) have no attribute either.
            if (auto convertedType = converter.convertType(bitsType)) {
                return IntegerAttr::get(convertedType, attr.getValue());
            }
            return LLVMTypeConverter::AttributeConversionResult::na();
        });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BoolType boolType, P4HIR::BoolAttr attr) {
            return IntegerAttr::get(converter.convertType(boolType), attr.getValue() ? 1 : 0);
        });
}

void P4::P4MLIR::populateP4HIRToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                       RewritePatternSet &patterns) {
    patterns.add<ConstOpConversion, BinOpConversion, UnaryOpConversion, CmpOpConversion,
                 ShlOpConversion, ShrOpConversion, ConcatOpConversion, SliceOpConversion>(
        converter);
}
