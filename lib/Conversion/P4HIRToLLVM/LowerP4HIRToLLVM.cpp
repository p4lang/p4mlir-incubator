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
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
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
        auto newAttr =
            getTypeConverter()->convertTypeAttribute(op.getValue().getType(), op.getValue());
        if (!newAttr) return rewriter.notifyMatchFailure(op, "unsupported constant type");
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, cast<TypedAttr>(*newAttr));
        return success();
    }
};

template <typename Op>
LogicalResult lowerToOp(P4HIR::BinOp op, P4HIR::BinOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<Op>(op, adaptor.getOperands());
    return success();
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
                return lowerToSignedOrUnsignedOp<LLVM::SDivOp, LLVM::UDivOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Mod:
                return lowerToSignedOrUnsignedOp<LLVM::SRemOp, LLVM::URemOp>(op, adaptor, rewriter);
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
        auto intType = dyn_cast<IntegerType>(input.getType());
        if (!intType) {
            return rewriter.notifyMatchFailure(op, "unsupported unary operand type");
        }

        auto createConstant = [&](int64_t value) {
            return LLVM::ConstantOp::create(rewriter, op.getLoc(), intType, value);
        };

        switch (op.getKind()) {
            case P4HIR::UnaryOpKind::UPlus:
                rewriter.replaceOp(op, input);
                return success();
            case P4HIR::UnaryOpKind::Neg:
                // LLVM has no integer negation, so `-x` is emitted as `0 - x`.
                rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, createConstant(0), input);
                return success();
            case P4HIR::UnaryOpKind::Cmpl:
                rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, input, createConstant(-1));
                return success();
            case P4HIR::UnaryOpKind::LNot:
                if (intType.getWidth() != 1) {
                    return rewriter.notifyMatchFailure(op, "logical not requires i1 operand");
                }
                rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, input, createConstant(1));
                return success();
        }
        return rewriter.notifyMatchFailure(op, "unsupported unary op kind");
    }
};

LogicalResult lowerToICmpOp(P4HIR::CmpOp op, P4HIR::CmpOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter, LLVM::ICmpPredicate predicate) {
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, predicate, adaptor.getLhs(), adaptor.getRhs());
    return success();
}

struct CmpOpConversion : public ConvertOpToLLVMPattern<P4HIR::CmpOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto lhsType = op.getLhs().getType();
        if (!isa<P4HIR::BitsType, P4HIR::BoolType>(lhsType)) {
            return rewriter.notifyMatchFailure(op, "unsupported cmp operand type");
        }

        bool isSigned = false;  // BoolType lowers to i1 and is always compared as unsigned.
        if (auto bitsType = dyn_cast<P4HIR::BitsType>(lhsType)) {
            isSigned = bitsType.isSigned();
        }

        switch (op.getKind()) {
            case P4HIR::CmpOpKind::Eq:
                return lowerToICmpOp(op, adaptor, rewriter, LLVM::ICmpPredicate::eq);
            case P4HIR::CmpOpKind::Ne:
                return lowerToICmpOp(op, adaptor, rewriter, LLVM::ICmpPredicate::ne);
            case P4HIR::CmpOpKind::Lt:
                return lowerToICmpOp(
                    op, adaptor, rewriter,
                    isSigned ? LLVM::ICmpPredicate::slt : LLVM::ICmpPredicate::ult);
            case P4HIR::CmpOpKind::Le:
                return lowerToICmpOp(
                    op, adaptor, rewriter,
                    isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule);
            case P4HIR::CmpOpKind::Gt:
                return lowerToICmpOp(
                    op, adaptor, rewriter,
                    isSigned ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::ugt);
            case P4HIR::CmpOpKind::Ge:
                return lowerToICmpOp(
                    op, adaptor, rewriter,
                    isSigned ? LLVM::ICmpPredicate::sge : LLVM::ICmpPredicate::uge);
        }
        return rewriter.notifyMatchFailure(op, "unsupported cmp op kind");
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

        if (failed(applyPartialConversion(module, target, std::move(patterns), config)))
            signalPassFailure();
    }
};

}  // namespace

void P4::P4MLIR::populateP4HIRToLLVMTypeConversion(LLVMTypeConverter &converter) {
    converter.addConversion([](P4HIR::BitsType bitsType) {
        return IntegerType::get(bitsType.getContext(), bitsType.getWidth());
    });

    converter.addConversion(
        [](P4HIR::BoolType boolType) { return IntegerType::get(boolType.getContext(), 1); });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BitsType bitsType, P4HIR::IntAttr attr) {
            return IntegerAttr::get(converter.convertType(bitsType), attr.getValue());
        });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BoolType boolType, P4HIR::BoolAttr attr) {
            return IntegerAttr::get(converter.convertType(boolType), attr.getValue() ? 1 : 0);
        });
}

void P4::P4MLIR::populateP4HIRToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                       RewritePatternSet &patterns) {
    patterns.add<ConstOpConversion, BinOpConversion, UnaryOpConversion, CmpOpConversion>(converter);
}
