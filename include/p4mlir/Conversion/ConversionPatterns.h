#ifndef P4MLIR_CONVERSION_CONVERSIONPATTERNS_H
#define P4MLIR_CONVERSION_CONVERSIONPATTERNS_H

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Transforms/DialectConversion.h"

namespace P4::P4MLIR {

/// A helper type converter class that automatically populates the relevant
///  type conversions for compound P4HIR types
class P4HIRTypeConverter : public mlir::TypeConverter {
 public:
    P4HIRTypeConverter();
};

// Performs type conversion on the given operation.
llvm::FailureOr<mlir::Operation *> doTypeConversion(mlir::Operation *op, mlir::ValueRange operands,
                                                    mlir::ConversionPatternRewriter &rewriter,
                                                    const mlir::TypeConverter *typeConverter);

/// Generic pattern which replaces an operation by one of the same operation
/// name, but with converted attributes, operands, and result types to eliminate
/// illegal types. Uses generic builders based on OperationState to make sure
/// that this pattern can apply to any operation.
///
/// Useful when a conversion can be entirely defined by a TypeConverter.
/// Normally TypeConverter will be either P4HIRTypeConverter or its descendant.
struct TypeConversionPattern : public mlir::ConversionPattern {
 public:
    TypeConversionPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context)
        : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}
    using ConversionPattern::ConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        return doTypeConversion(op, operands, rewriter, getTypeConverter());
    }
};

// Specialization of the above which targets a specific operation.
template <typename OpTy>
struct TypeOpConversionPattern : public mlir::OpConversionPattern<OpTy> {
    using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        return doTypeConversion(op.getOperation(), adaptor.getOperands(), rewriter,
                                this->getTypeConverter());
    }
};

template <typename... OpTypes>
void populateTypeOpConversionPattern(mlir::RewritePatternSet &patterns,
                                     const mlir::TypeConverter &converter) {
    (patterns.add<TypeOpConversionPattern<OpTypes>>(converter, patterns.getContext()), ...);
}

template <typename... OpTypes>
void populateP4HIRFunctionOpTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                  const mlir::TypeConverter &converter) {
    (patterns.add<TypeOpConversionPattern<OpTypes>>(converter, patterns.getContext()), ...);
}

void populateP4HIRAnyCallOpTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                 const mlir::TypeConverter &converter);

}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_CONVERSIONPATTERNS_H
