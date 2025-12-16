#include "p4mlir/Conversion/P4HIRToCoreLib/P4HIRToCoreLib.h"

#include <optional>

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define DEBUG_TYPE "p4hir-convert-to-corelib"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERTOP4CORELIB
#include "p4mlir/Conversion/P4HIRToCoreLib/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct LowerToP4CoreLib : public P4::P4MLIR::impl::LowerToP4CoreLibBase<LowerToP4CoreLib> {
    void runOnOperation() override;
};

// TODO: Patterns here are very straightforward, switch to PDLL

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

struct OverloadSetOpConversionPattern : public OpConversionPattern<P4HIR::OverloadSetOp> {
    using OpConversionPattern<P4HIR::OverloadSetOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::OverloadSetOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // Just remove the op, all calls to it should already be legalized
        rewriter.eraseOp(op);

        return success();
    }
};

struct ExternOpConversionPattern : public OpConversionPattern<P4HIR::ExternOp> {
    using OpConversionPattern<P4HIR::ExternOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ExternOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
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
        if (callee.getLeafReference() == "verify") {
            rewriter.replaceOpWithNewOp<P4CoreLib::VerifyOp>(op, mlir::TypeRange(),
                                                             operands.getArgOperands());
            return success();
        }

        return failure();
    }
};

struct CallMethodOpConversionPattern : public OpConversionPattern<P4HIR::CallMethodOp> {
    using OpConversionPattern<P4HIR::CallMethodOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::CallMethodOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sym = op.getMethod();
        auto extSym = sym.getRootReference(), methodSym = sym.getLeafReference();
        if (extSym == "packet_out" && methodSym == "emit") {
            rewriter.replaceOpWithNewOp<P4CoreLib::PacketEmitOp>(op, mlir::TypeRange(),
                                                                 operands.getOperands());
            return success();
        } else if (extSym == "packet_in") {
            if (methodSym == "extract") {
                size_t sz = op.getArgOperands().size();
                if (sz == 1) {
                    rewriter.replaceOpWithNewOp<P4CoreLib::PacketExtractOp>(op, op.getResultTypes(),
                                                                            operands.getOperands());
                    return success();
                } else if (sz == 2) {
                    rewriter.replaceOpWithNewOp<P4CoreLib::PacketExtractVariableOp>(
                        op, op.getResultTypes(), operands.getOperands());
                    return success();
                }
            } else if (methodSym == "length") {
                rewriter.replaceOpWithNewOp<P4CoreLib::PacketLengthOp>(op, op.getResultTypes(),
                                                                       operands.getOperands());
                return success();
            } else if (methodSym == "lookahead") {
                rewriter.replaceOpWithNewOp<P4CoreLib::PacketLookAheadOp>(op, op.getResultTypes(),
                                                                          operands.getOperands());
                return success();
            } else if (methodSym == "advance") {
                rewriter.replaceOpWithNewOp<P4CoreLib::PacketAdvanceOp>(op, op.getResultTypes(),
                                                                        operands.getOperands());
                return success();
            }
        }

        return failure();
    }
};

void LowerToP4CoreLib::runOnOperation() {
    MLIRContext &context = getContext();
    mlir::ModuleOp module = getOperation();

    ConversionTarget target(context);
    RewritePatternSet patterns(&context);
    SymbolTableCollection symTables;

    P4HIRTypeConverter converter;
    converter.addConversion([&](P4HIR::ExternType extType) -> std::optional<Type> {
        if (!extType.hasAnnotation("corelib")) return std::nullopt;

        if (extType.getName() == "packet_in")
            return P4CoreLib::PacketInType::get(extType.getContext());
        else if (extType.getName() == "packet_out")
            return P4CoreLib::PacketOutType::get(extType.getContext());

        return std::nullopt;
    });

    target.addLegalDialect<P4CoreLib::P4CoreLibDialect>();

    target.addDynamicallyLegalOp<P4HIR::FuncOp>([&](P4HIR::FuncOp func) {
        // All corelib-annotated functions should be converted
        return !func.hasAnnotation("corelib") && converter.isLegal(func.getFunctionType());
    });
    target.addDynamicallyLegalOp<P4HIR::ExternOp>([](P4HIR::ExternOp ext) {
        // All corelib-annotated externs should be converted
        return !ext.hasAnnotation("corelib");
    });

    target.addDynamicallyLegalOp<P4HIR::CallOp>([&](P4HIR::CallOp call) {
        // All calls to corelib-annotated callees should be converted
        // Check the callee, must be a corelib extern
        auto *calleeOp = call.resolveCallableInTable(&symTables);
        if (auto opFunc = dyn_cast_or_null<P4HIR::FuncOp>(calleeOp))
            return !target.isIllegal(opFunc);

        assert(calleeOp && isa<FunctionOpInterface>(calleeOp));
        return true;
    });

    target.addDynamicallyLegalOp<P4HIR::CallMethodOp>([&](P4HIR::CallMethodOp call) {
        // All calls to methods of corelib-annotated externs should be converted
        return !call.getExtern().hasAnnotation("corelib");
    });

    target.addDynamicallyLegalOp<P4HIR::OverloadSetOp>([&](P4HIR::OverloadSetOp ovl) {
        for (auto opFunc : ovl.getOps<P4HIR::FuncOp>()) return !target.isIllegal(opFunc);
        return true;
    });

    configureUnknownOpDynamicallyLegalByTypes(target, converter);

    patterns.add<FuncOpConversionPattern, ExternOpConversionPattern, CallOpConversionPattern,
                 CallMethodOpConversionPattern, OverloadSetOpConversionPattern>(converter,
                                                                                &context);

    populateTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

}  // end anonymous namespace
