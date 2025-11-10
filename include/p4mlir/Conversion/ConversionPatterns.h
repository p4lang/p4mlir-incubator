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
    TypeConversionPattern(const mlir::TypeConverter &converter, mlir::MLIRContext *context)
        : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}
    using ConversionPattern::ConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        return doTypeConversion(op, operands, rewriter, getTypeConverter());
    }
};

// Specialization of `TypeConversionPattern` which targets a specific operation.
template <typename OpTy>
struct OpTypeConversionPattern : public mlir::OpConversionPattern<OpTy> {
    using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        return doTypeConversion(op.getOperation(), adaptor.getOperands(), rewriter,
                                this->getTypeConverter());
    }
};

// Specialization of `TypeConversionPattern` which targets a specific op interface.
template <typename OpInterfaceTy>
struct OpInterfaceTypeConversionPattern : public mlir::OpInterfaceConversionPattern<OpInterfaceTy> {
    using mlir::OpInterfaceConversionPattern<OpInterfaceTy>::OpInterfaceConversionPattern;

    mlir::LogicalResult matchAndRewrite(OpInterfaceTy op, llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        return doTypeConversion(op, operands, rewriter, this->getTypeConverter());
    }
};

inline void populateTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                          const mlir::TypeConverter &converter) {
    patterns.addWithLabel<TypeConversionPattern>({"generic type converter"}, converter,
                                                 patterns.getContext());
}

template <typename... OpTypes>
inline void populateOpTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                            const mlir::TypeConverter &converter) {
    (patterns.addWithLabel<OpTypeConversionPattern<OpTypes>>(
         {"op type conversion", OpTypes::getOperationName()}, converter, patterns.getContext()),
     ...);
}

template <typename... OpTypes>
inline void populateOpInterfaceTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                     const mlir::TypeConverter &converter) {
    (patterns.addWithLabel<OpInterfaceTypeConversionPattern<OpTypes>>(
         {"interface op type conversion"}, converter, patterns.getContext()),
     ...);
}

// Mark unknown operations legal if their types and attributes are valid for `converter`.
void configureUnknownOpDynamicallyLegalByTypes(mlir::ConversionTarget &target,
                                               const mlir::TypeConverter &converter);

}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_CONVERSIONPATTERNS_H
