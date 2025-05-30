#include "p4mlir/Conversion/P4HIRToCoreLib.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "p4hir-convert-to-corelib"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERTOP4CORELIB
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct LowerToP4CoreLib : public P4::P4MLIR::impl::LowerToP4CoreLibBase<LowerToP4CoreLib> {
    void runOnOperation() override;
};

struct FuncOpConversionPattern : public OpConversionPattern<P4HIR::FuncOp> {
    using OpConversionPattern<P4HIR::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::FuncOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op.isDeclaration()) return failure();

        // Just remove the op, all calls to it should already be legalized
        rewriter.eraseOp(op);

        return success();
    }
};

struct CallOpConversionPattern : public OpConversionPattern<P4HIR::CallOp> {
    using OpConversionPattern<P4HIR::CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::CallOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto callee = op.getCallee();
        if (callee == "verify") {
            rewriter.replaceOpWithNewOp<P4CoreLib::VerifyOp>(op, mlir::TypeRange(),
                                                             operands.getArgOperands());
            return success();
        }

        return failure();
    }
};

}  // end anonymous namespace

void LowerToP4CoreLib::runOnOperation() {
    MLIRContext &context = getContext();
    mlir::ModuleOp module = getOperation();

    ConversionTarget target(context);
    RewritePatternSet patterns(&context);
    SymbolTableCollection symTables;

    target.addLegalDialect<P4CoreLib::P4CoreLibDialect>();

    target.addDynamicallyLegalOp<P4HIR::FuncOp>([](P4HIR::FuncOp func) {
        // All corelib-annotated functions should be converted
        return !func.hasAnnotation("corelib");
    });
    target.addDynamicallyLegalOp<P4HIR::CallOp>([&](P4HIR::CallOp call) {
        // All calls to corelib-annotated callees should be converted
        // Check the callee, must be a corelib extern
        auto *calleeOp = call.resolveCallableInTable(&symTables);
        if (auto opFunc = dyn_cast_or_null<P4HIR::FuncOp>(calleeOp)) {
            if (!opFunc.isDeclaration()) return true;
            return !opFunc.hasAnnotation("corelib");
        }

        assert(calleeOp && isa<FunctionOpInterface>(calleeOp));
        return true;
    });

    patterns.add<FuncOpConversionPattern, CallOpConversionPattern>(&context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}
