#include "p4mlir/Transforms/DialectConversion.h"

#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "p4hir-dialect-conversion"

using namespace mlir;

using namespace P4::P4MLIR;

namespace P4::P4MLIR::utils {

llvm::LogicalResult FunctionOpInterfaceConversionPattern::matchAndRewrite(
    mlir::FunctionOpInterface funcOp, llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
    if (failed(convertFuncOpTypes(funcOp, *typeConverter, rewriter))) return mlir::failure();

    if (failed(convertCtorTypes(funcOp.getOperation(), *typeConverter, rewriter)))
        return mlir::failure();

    return mlir::success();
}

LogicalResult convertFuncOpTypes(FunctionOpInterface funcOp, const TypeConverter &typeConverter,
                                 ConversionPatternRewriter &rewriter) {
    auto fnType = mlir::dyn_cast<P4HIR::FuncType>(funcOp.getFunctionType());
    if (!fnType) {
        return mlir::failure();
    }

    TypeConverter::SignatureConversion result(fnType.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter.convertSignatureArgs(fnType.getInputs(), result)) ||
        failed(typeConverter.convertTypes(fnType.getReturnTypes(), newResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(), typeConverter, &result)))
        return mlir::failure();

    // Update the function signature in-place.
    auto newType = P4HIR::FuncType::get(rewriter.getContext(), result.getConvertedTypes(),
                                        newResults.empty() ? mlir::Type() : newResults.front(),
                                        fnType.getTypeArguments());
    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });

    return mlir::success();
}

llvm::LogicalResult convertCtorTypes(mlir::Operation *op, const mlir::TypeConverter &typeConverter,
                                     mlir::ConversionPatternRewriter &rewriter) {
    if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
        auto ctorType = parserOp.getCtorType();
        if (!typeConverter.isLegal(ctorType.getReturnType())) {
            // Expect empty ctor args
            if (ctorType.getNumInputs())
                return rewriter.notifyMatchFailure(parserOp, "non-empty inputs for ctor types");
            auto newType =
                P4HIR::CtorType::get(rewriter.getContext(), ctorType.getInputs(),
                                     typeConverter.convertType(ctorType.getReturnType()));
            rewriter.modifyOpInPlace(parserOp, [&] { parserOp.setCtorType(newType); });
        }
        return mlir::success();
    }

    if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
        auto ctorType = controlOp.getCtorType();
        if (!typeConverter.isLegal(ctorType.getReturnType())) {
            // Expect empty ctor args
            if (ctorType.getNumInputs())
                return rewriter.notifyMatchFailure(controlOp, "non-empty inputs for ctor types");
            auto newType =
                P4HIR::CtorType::get(rewriter.getContext(), ctorType.getInputs(),
                                     typeConverter.convertType(ctorType.getReturnType()));
            rewriter.modifyOpInPlace(controlOp, [&] { controlOp.setCtorType(newType); });
        }
        return mlir::success();
    }

    // Not a ctor type op, nothing to do
    return mlir::success();
}

}  // namespace P4::P4MLIR::utils
