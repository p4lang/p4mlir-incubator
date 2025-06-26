#ifndef P4MLIR_TRANSFORMS_DIALECTCONVERSION_H_
#define P4MLIR_TRANSFORMS_DIALECTCONVERSION_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace P4::P4MLIR {

llvm::LogicalResult convertFuncOpTypes(mlir::FunctionOpInterface funcOp,
                                       const mlir::TypeConverter &typeConverter,
                                       mlir::ConversionPatternRewriter &rewriter);

llvm::LogicalResult convertCallOpTypes(mlir::CallOpInterface callOp,
                                       llvm::ArrayRef<mlir::Value> operands,
                                       const mlir::TypeConverter &typeConverter,
                                       mlir::ConversionPatternRewriter &rewriter);

void populateFunctionOpInterfaceTypeConversionPattern(mlir::StringRef functionLikeOpName,
                                                      mlir::RewritePatternSet &patterns,
                                                      const mlir::TypeConverter &converter);

void populateCallOpInterfaceTypeConversionPattern(mlir::StringRef callLikeOpName,
                                                  mlir::RewritePatternSet &patterns,
                                                  const mlir::TypeConverter &converter);

template <typename... FuncOpTs>
void populateFunctionOpInterfaceTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                      const mlir::TypeConverter &converter) {
    (P4::P4MLIR::populateFunctionOpInterfaceTypeConversionPattern(FuncOpTs::getOperationName(),
                                                                  patterns, converter),
     ...);
}

template <typename... CallOpTs>
void populateCallOpInterfaceTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                  const mlir::TypeConverter &converter) {
    (P4::P4MLIR::populateCallOpInterfaceTypeConversionPattern(CallOpTs::getOperationName(),
                                                              patterns, converter),
     ...);
}

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_DIALECTCONVERSION_H
