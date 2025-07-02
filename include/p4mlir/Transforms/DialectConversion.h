#ifndef P4MLIR_TRANSFORMS_DIALECTCONVERSION_H_
#define P4MLIR_TRANSFORMS_DIALECTCONVERSION_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace P4::P4MLIR::utils {

llvm::LogicalResult convertFuncOpTypes(mlir::FunctionOpInterface funcOp,
                                       const mlir::TypeConverter &typeConverter,
                                       mlir::ConversionPatternRewriter &rewriter);

llvm::LogicalResult convertCtorTypes(mlir::Operation* op,
                                     const mlir::TypeConverter &typeConverter,
                                     mlir::ConversionPatternRewriter &rewriter);

struct FunctionOpInterfaceConversionPattern
    : public mlir::OpInterfaceConversionPattern<mlir::FunctionOpInterface> {
    using OpInterfaceConversionPattern<mlir::FunctionOpInterface>::OpInterfaceConversionPattern;
    llvm::LogicalResult matchAndRewrite(mlir::FunctionOpInterface funcOp,
                                        llvm::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename OpTy>
class GenericOpTypeConversionPattern : public mlir::OpConversionPattern<OpTy> {
 public:
    using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::FailureOr<mlir::Operation *> newOp =
            convertOpResultTypes(op, adaptor.getOperands(), *this->getTypeConverter(), rewriter);
        if (failed(newOp)) return mlir::failure();

        rewriter.replaceOp(op, (*newOp)->getResults());
        return mlir::success();
    }
};

template <typename... OpTypes>
void populateGenericOpTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                            const mlir::TypeConverter &converter) {
    (patterns.add<P4::P4MLIR::utils::GenericOpTypeConversionPattern<OpTypes>>(
         converter, patterns.getContext()),
     ...);
}

}  // namespace P4::P4MLIR::utils

#endif  // P4MLIR_TRANSFORMS_DIALECTCONVERSION_H
