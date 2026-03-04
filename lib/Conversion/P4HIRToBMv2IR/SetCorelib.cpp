#include <array>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "set-corelib"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SETCORELIB
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
bool isCoreLibName(StringRef name) {
    const std::array<StringRef, 4> names{"packet_out", "emit", "packet_in", "extract"};
    return llvm::is_contained(names, name);
}
struct SetCoreLibTypeConverter : TypeConverter {
    SetCoreLibTypeConverter() {
        addConversion([](Type ty) { return ty; });
        addConversion([](P4HIR::ExternType ty) {
            if (ty.hasAnnotation("corelib")) return ty;
            auto name = ty.getName();
            if (!isCoreLibName(name)) return ty;
            auto annotations = ty.getAnnotations();
            SmallVector<NamedAttribute> newAttrs;
            if (annotations) newAttrs.append(annotations.begin(), annotations.end());
            newAttrs.emplace_back("corelib", BoolAttr::get(ty.getContext(), true));
            return P4HIR::ExternType::get(ty.getContext(), name,
                                          DictionaryAttr::get(ty.getContext(), newAttrs));
        });
        addConversion([&](P4HIR::FuncType funcTy) {
            SmallVector<Type> convertedInputs;
            SmallVector<Type> convertedTypeArgs;

            for (auto in : funcTy.getInputs()) convertedInputs.push_back(convertType(in));
            for (auto in : funcTy.getTypeArguments()) convertedTypeArgs.push_back(convertType(in));
            auto convertedOut = convertType(funcTy.getReturnType());
            return P4HIR::FuncType::get(convertedInputs, convertedOut, convertedTypeArgs);
        });
    }
};

struct SetCoreLibPattern : public ConversionPattern {
    SetCoreLibPattern(SetCoreLibTypeConverter &typeConverter, MLIRContext *context)
        : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sym = dyn_cast<SymbolOpInterface>(op);
        if (!sym) return failure();
        auto symName = sym.getName();
        if (!isCoreLibName(symName)) return failure();

        auto trueAttr = BoolAttr::get(sym.getContext(), true);
        NamedAttribute coreLibAttr("corelib", trueAttr);
        auto corelibAdded = llvm::TypeSwitch<Operation *, LogicalResult>(sym.getOperation())
                                .Case([&](P4HIR::FuncOp funcOp) {
                                    funcOp.setAnnotation("corelib", trueAttr);
                                    return success();
                                })
                                .Case([&](P4HIR::ExternOp externOp) {
                                    externOp.setAnnotation("corelib", trueAttr);
                                    return success();
                                })
                                .Case([&](P4HIR::OverloadSetOp overloadSetOp) {
                                    overloadSetOp.walk([&](P4HIR::FuncOp overloadFunc) {
                                        overloadFunc.setAnnotation("corelib", trueAttr);
                                    });
                                    return success();
                                })
                                .Default([](Operation *) { return failure(); });

        if (failed(corelibAdded)) return failure();
        auto typeConverted =
            doTypeConversion(sym.getOperation(), operands, rewriter, getTypeConverter());
        if (failed(typeConverted)) return failure();
        return success();
    }
};

struct SetCorelibPass : public P4::P4MLIR::impl::SetCorelibBase<SetCorelibPass> {
    void runOnOperation() override {
        auto moduleOp = getOperation();
        auto &context = getContext();
        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        SetCoreLibTypeConverter converter;

        target.addDynamicallyLegalOp<P4HIR::OverloadSetOp>([](P4HIR::OverloadSetOp overloadSetOp) {
            if (!isCoreLibName(overloadSetOp.getSymName())) return true;
            bool allCore = true;
            overloadSetOp.walk(
                [&](P4HIR::FuncOp funcOp) { allCore &= funcOp.hasAnnotation("corelib"); });
            return allCore;
        });
        target.addDynamicallyLegalOp<P4HIR::ExternOp>([](P4HIR::ExternOp op) {
            return !isCoreLibName(op.getName()) || op.hasAnnotation("corelib");
        });
        target.addDynamicallyLegalOp<P4HIR::FuncOp>([](P4HIR::FuncOp op) {
            return !isCoreLibName(op.getName()) || op.hasAnnotation("corelib");
        });
        configureUnknownOpDynamicallyLegalByTypes(target, converter);

        patterns.add<SetCoreLibPattern>(converter, &context);
        populateTypeConversionPattern(patterns, converter);
        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
            signalPassFailure();
    };
};

}  // anonymous namespace
