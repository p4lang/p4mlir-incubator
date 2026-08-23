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
#include "p4mlir/Dialect/P4HIR/Matchers.h"
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

LogicalResult diagnoseDivisionByZero(ModuleOp module) {
    auto walkResult = module.walk([](P4HIR::BinOp op) -> WalkResult {
        auto kind = op.getKind();
        if (kind != P4HIR::BinOpKind::Div && kind != P4HIR::BinOpKind::Mod) {
            return WalkResult::advance();
        }
        if (!mlir::matchPattern(op.getRhs(), m_ZeroInt())) {
            return WalkResult::advance();
        }
        return op.emitError(kind == P4HIR::BinOpKind::Div ? "division by zero"
                                                          : "modulo by zero");
    });
    return failure(walkResult.wasInterrupted());
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

struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    void runOnOperation() override {
        auto &context = getContext();
        auto module = getOperation();

        if (failed(diagnoseDivisionByZero(module))) {
            signalPassFailure();
            return;
        }

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
        if (bitsType.getWidth() == 0) {
            return std::nullopt;
        }
        return IntegerType::get(bitsType.getContext(), bitsType.getWidth());
    });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BitsType bitsType,
                     P4HIR::IntAttr attr) -> LLVMTypeConverter::AttributeConversionResult {
            // Types without an LLVM counterpart (e.g. `bit<0>`) have no attribute either.
            auto convertedType = converter.convertType(bitsType);
            if (!convertedType) {
                return LLVMTypeConverter::AttributeConversionResult::na();
            }
            return IntegerAttr::get(convertedType, attr.getValue());
        });
}

void P4::P4MLIR::populateP4HIRToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                       RewritePatternSet &patterns) {
    patterns.add<ConstOpConversion, BinOpConversion>(converter);
}
