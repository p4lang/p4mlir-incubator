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

// Converts a successor block's argument types to match the converted branch
// operands. Entry blocks have no predecessors; their signatures are converted
// when lowering the operation that owns the region.
// Follows the upstream ControlFlowToLLVM lowering.
FailureOr<Block *> getConvertedBlock(ConversionPatternRewriter &rewriter,
                                     const TypeConverter *converter, Operation *branchOp,
                                     Block *block, TypeRange expectedTypes) {
    assert(!block->isEntryBlock() && "entry blocks have no predecessors");

    // There is nothing to do if the types already match, e.g. if the block was already converted
    // for another predecessor.
    if (block->getArgumentTypes() == expectedTypes) return block;

    auto conversion = converter->convertBlockSignature(block);
    if (!conversion)
        return rewriter.notifyMatchFailure(branchOp, "could not compute block signature");
    if (expectedTypes != conversion->getConvertedTypes())
        return rewriter.notifyMatchFailure(branchOp,
                                           "block signature does not match branch operands");
    return rewriter.applySignatureConversion(block, *conversion, converter);
}

struct BrOpConversion : public ConvertOpToLLVMPattern<P4HIR::BrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::BrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto dest = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDest(),
                                      TypeRange(adaptor.getDestOperands()));
        if (failed(dest)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, adaptor.getDestOperands(), *dest);
        return success();
    }
};

struct CondBrOpConversion : public ConvertOpToLLVMPattern<P4HIR::CondBrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CondBrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto destTrue = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestTrue(),
                                          TypeRange(adaptor.getDestOperandsTrue()));
        if (failed(destTrue)) return failure();
        auto destFalse = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestFalse(),
                                           TypeRange(adaptor.getDestOperandsFalse()));
        if (failed(destFalse)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, adaptor.getCond(), *destTrue,
                                                    adaptor.getDestOperandsTrue(), *destFalse,
                                                    adaptor.getDestOperandsFalse());
        return success();
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

// Returns the LLVM constant for the default value of `type`, or null if it has none.
TypedAttr getDefaultValueAttr(Type type, const TypeConverter &converter) {
    if (auto defaultValueType = dyn_cast<P4HIR::HasDefaultValue>(type))
        if (auto defaultValue = defaultValueType.getDefaultValue())
            return dyn_cast_if_present<TypedAttr>(
                converter.convertTypeAttribute(type, defaultValue).value_or(Attribute()));
    return {};
}

// P4 leaves the value of an uninitialized variable unspecified. Like mem2reg on P4HIR, the
// lowering takes the default value of its type, rather than leaving the alloca undefined, unless
// `initializeVariables` is off.
struct VariableOpConversion : public ConvertOpToLLVMPattern<P4HIR::VariableOp> {
    VariableOpConversion(const LLVMTypeConverter &converter, bool initializeVariables)
        : ConvertOpToLLVMPattern(converter), initializeVariables(initializeVariables) {}

    LogicalResult matchAndRewrite(P4HIR::VariableOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!isa_and_present<P4HIR::FuncOp>(op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()))
            return rewriter.notifyMatchFailure(op, "not in the body of a function");

        auto objectType = getTypeConverter()->convertType(op.getObjectType());
        if (!objectType) return rewriter.notifyMatchFailure(op, "unsupported object type");

        // A variable with `init` is initialized by its first use. One with a marked lifetime
        // takes its default value where its lifetime starts.
        TypedAttr defaultValue;
        if (initializeVariables && !op.getInit() &&
            llvm::none_of(op->getUsers(), llvm::IsaPred<P4HIR::LifetimeStartOp>)) {
            defaultValue = getDefaultValueAttr(op.getObjectType(), *getTypeConverter());
            if (!defaultValue) return rewriter.notifyMatchFailure(op, "no default value");
        }

        auto loc = op.getLoc();
        auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(), 1);
        auto alloca = LLVM::AllocaOp::create(rewriter, loc, getPtrType(), objectType, one);
        if (defaultValue)
            LLVM::StoreOp::create(rewriter, loc,
                                  LLVM::ConstantOp::create(rewriter, loc, defaultValue), alloca);
        rewriter.replaceOp(op, alloca);
        return success();
    }

    // Whether to store the default value into variables without an initializer.
    bool initializeVariables;
};

// LLVM only accepts lifetime markers on allocas: those of variables that are not lowered stay.
FailureOr<LLVM::AllocaOp> getLoweredVariable(Operation *op, Value ref,
                                             ConversionPatternRewriter &rewriter) {
    if (auto alloca = ref.getDefiningOp<LLVM::AllocaOp>()) return alloca;
    return rewriter.notifyMatchFailure(op, "variable is not lowered");
}

// A variable comes into existence holding the default value of its type.
struct LifetimeStartOpConversion : public ConvertOpToLLVMPattern<P4HIR::LifetimeStartOp> {
    LifetimeStartOpConversion(const LLVMTypeConverter &converter, bool initializeVariables)
        : ConvertOpToLLVMPattern(converter), initializeVariables(initializeVariables) {}


    LogicalResult matchAndRewrite(P4HIR::LifetimeStartOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto alloca = getLoweredVariable(op, adaptor.getRef(), rewriter);
        if (failed(alloca)) return failure();

        // A variable with `init` is initialized by its first use.
        auto variable = op.getRef().getDefiningOp<P4HIR::VariableOp>();
        TypedAttr defaultValue;
        if (initializeVariables && (!variable || !variable.getInit())) {
            auto objectType = cast<P4HIR::ReferenceType>(op.getRef().getType()).getObjectType();
            defaultValue = getDefaultValueAttr(objectType, *getTypeConverter());
            if (!defaultValue) return rewriter.notifyMatchFailure(op, "no default value");
        }

        auto loc = op.getLoc();
        LLVM::LifetimeStartOp::create(rewriter, loc, *alloca);
        if (defaultValue)
            LLVM::StoreOp::create(rewriter, loc,
                                  LLVM::ConstantOp::create(rewriter, loc, defaultValue), *alloca);
        rewriter.eraseOp(op);
        return success();
    }
    // Whether to store the default value into variables without an initializer.
    bool initializeVariables;
};

struct LifetimeEndOpConversion : public ConvertOpToLLVMPattern<P4HIR::LifetimeEndOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::LifetimeEndOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto alloca = getLoweredVariable(op, adaptor.getRef(), rewriter);
        if (failed(alloca)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::LifetimeEndOp>(op, *alloca);
        return success();
    }
};

struct ReadOpConversion : public ConvertOpToLLVMPattern<P4HIR::ReadOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ReadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto resultType = getTypeConverter()->convertType(op.getType());
        if (!resultType) return rewriter.notifyMatchFailure(op, "unsupported result type");

        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, adaptor.getRef());
        return success();
    }
};

struct AssignOpConversion : public ConvertOpToLLVMPattern<P4HIR::AssignOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), adaptor.getRef());
        return success();
    }
};

struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    using LowerP4HIRToLLVMBase::LowerP4HIRToLLVMBase;
    void runOnOperation() override {
        auto &context = getContext();
        auto module = getOperation();

        LLVMTypeConverter typeConverter(&context);
        populateP4HIRToLLVMTypeConversion(typeConverter);

        LLVMConversionTarget target(context);
        target.addLegalOp<ModuleOp>();

        RewritePatternSet patterns(&context);
        populateP4HIRToLLVMConversionPatterns(typeConverter, patterns, initializeVariables);

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

    // References lower to opaque pointers, as the memory operations carry the object type.
    // A reference to an object without an LLVM counterpart has none either.
    converter.addConversion([&converter](P4HIR::ReferenceType refType) -> std::optional<Type> {
        if (!converter.convertType(refType.getObjectType())) return std::nullopt;
        return LLVM::LLVMPointerType::get(refType.getContext());
    });

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
                                                       RewritePatternSet &patterns,
                                                       bool initializeVariables) {
    patterns.add<ConstOpConversion, BinOpConversion, UnaryOpConversion, CmpOpConversion,
                 ConcatOpConversion, SliceOpConversion>(converter);
    patterns.add<BrOpConversion, CondBrOpConversion>(converter);
    patterns.add<VariableOpConversion, LifetimeStartOpConversion>(converter, initializeVariables);
    patterns.add<ReadOpConversion, AssignOpConversion, LifetimeEndOpConversion>(converter);
}
