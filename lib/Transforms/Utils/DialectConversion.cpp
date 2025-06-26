#include "p4mlir/Transforms/DialectConversion.h"

#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "p4hir-dialect-conversion"

using namespace mlir;

namespace P4::P4MLIR {

//===----------------------------------------------------------------------===//
// FunctionOpInterface Conversions
//===----------------------------------------------------------------------===//

LogicalResult convertFuncOpTypes(FunctionOpInterface funcOp, const TypeConverter &typeConverter,
                                 ConversionPatternRewriter &rewriter) {
    auto fnType = mlir::dyn_cast<P4HIR::FuncType>(funcOp.getFunctionType());
    if (!fnType) {
        return failure();
    }

    TypeConverter::SignatureConversion result(fnType.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter.convertSignatureArgs(fnType.getInputs(), result)) ||
        failed(typeConverter.convertTypes(fnType.getReturnTypes(), newResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(), typeConverter, &result)))
        return failure();

    // Update the function signature in-place.
    auto newType = P4HIR::FuncType::get(rewriter.getContext(), result.getConvertedTypes(),
                                        newResults.empty() ? mlir::Type() : newResults.front(),
                                        fnType.getTypeArguments());
    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });

    return success();
}

struct FunctionOpInterfaceConversionPattern : public ConversionPattern {
    FunctionOpInterfaceConversionPattern(StringRef functionLikeOpName, MLIRContext *ctx,
                                         const TypeConverter &converter)
        : ConversionPattern(converter, functionLikeOpName, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
        return convertFuncOpTypes(funcOp, *typeConverter, rewriter);
    }
};

void populateFunctionOpInterfaceTypeConversionPattern(StringRef functionLikeOpName,
                                                      RewritePatternSet &patterns,
                                                      const TypeConverter &converter) {
    patterns.add<FunctionOpInterfaceConversionPattern>(functionLikeOpName, patterns.getContext(),
                                                       converter);
}

//===----------------------------------------------------------------------===//
// CallOpInterface Conversions
//===----------------------------------------------------------------------===//

LogicalResult convertCallOpTypes(CallOpInterface callOp, ArrayRef<Value> operands,
                                 const TypeConverter &typeConverter,
                                 ConversionPatternRewriter &rewriter) {
    FailureOr<Operation *> newOp = convertOpResultTypes(callOp, operands, typeConverter, rewriter);
    if (failed(newOp)) return failure();

    rewriter.replaceOp(callOp, (*newOp)->getResults());
    return success();
}

struct CallOpInterfaceConversionPattern : public ConversionPattern {
    CallOpInterfaceConversionPattern(StringRef callLikeOpName, MLIRContext *ctx,
                                     const TypeConverter &converter)
        : ConversionPattern(converter, callLikeOpName, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        CallOpInterface callOp = cast<CallOpInterface>(op);
        return convertCallOpTypes(callOp, operands, *typeConverter, rewriter);
    }
};

void populateCallOpInterfaceTypeConversionPattern(StringRef callLikeOpName,
                                                  RewritePatternSet &patterns,
                                                  const TypeConverter &converter) {
    patterns.add<CallOpInterfaceConversionPattern>(callLikeOpName, patterns.getContext(),
                                                   converter);
}

}  // namespace P4::P4MLIR
