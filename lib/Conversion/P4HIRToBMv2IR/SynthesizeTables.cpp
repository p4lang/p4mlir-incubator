#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "synthesize-tables"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SYNTHESIZETABLES
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct SynthTableFromActionCall : public OpRewritePattern<P4HIR::CallOp> {
    using OpRewritePattern<P4HIR::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::CallOp callOp, PatternRewriter &rewriter) const override {
        auto controlApplyParent = callOp->getParentOfType<P4HIR::ControlApplyOp>();
        if (!controlApplyParent) return failure();
        auto controlParent = controlApplyParent->getParentOfType<P4HIR::ControlOp>();
        if (!controlParent) return failure();
        if (callOp.getNumResults() != 0)
            return callOp.emitError("Can't replace action with results");
        if (callOp.getNumOperands() != 0) return callOp.emitError("Replacing action with args NYI");

        auto newTable = getTableForActionCall(callOp, controlApplyParent, controlParent, rewriter);
        auto refAttr = SymbolRefAttr::get(controlParent.getSymNameAttr(),
                                          {SymbolRefAttr::get(newTable.getSymNameAttr())});
        auto voidTy = P4HIR::StructType::get(callOp.getContext(), "dummy_apply_res", {});
        rewriter.create<P4HIR::TableApplyOp>(callOp.getLoc(), voidTy, refAttr, ValueRange{},
                                             nullptr);
        // We can't directly replace the call because the table_apply op returns the result of the
        // apply, while the action returns nothing
        rewriter.eraseOp(callOp);

        return success();
    }

 private:
    static P4HIR::TableOp getTableForActionCall(P4HIR::CallOp callOp,
                                                P4HIR::ControlApplyOp controlApplyParent,
                                                P4HIR::ControlOp controlParent,
                                                PatternRewriter &rewriter) {
        auto name = rewriter.getStringAttr("dummy_table");
        unsigned counter = 0;
        auto uniqueName = SymbolTable::generateSymbolName<256>(
            name,
            [&](StringRef candidate) {
                return SymbolTable::lookupSymbolIn(controlParent, candidate) != nullptr;
            },
            counter);
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(controlApplyParent);
        auto newTable = rewriter.create<P4HIR::TableOp>(callOp.getLoc(), uniqueName, nullptr);
        auto &block = newTable.getBody().emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        auto funcTy = P4HIR::FuncType::get(callOp.getContext(), {});
        rewriter.create<P4HIR::TableActionOp>(
            callOp.getLoc(), SymbolRefAttr::get(callOp.getCalleeAttr().getLeafReference()), funcTy,
            ArrayRef<DictionaryAttr>{}, nullptr,
            [&](OpBuilder &builder, Block::BlockArgListType args, Location loc) {
                builder.create<P4HIR::CallOp>(loc, callOp.getCalleeAttr(), args);
            });

        const bool defaultActionConst = true;
        rewriter.create<P4HIR::TableDefaultActionOp>(
            callOp.getLoc(), defaultActionConst, nullptr, [&](OpBuilder &builder, Location loc) {
                builder.create<P4HIR::CallOp>(loc, callOp.getCalleeAttr(), ValueRange{});
            });

        constexpr unsigned defaultTableSize =
            1024;  // Same default size as p4c, see backends/bmv2/common/helpers.cpp
        auto infIntTy = P4HIR::InfIntType::get(rewriter.getContext());
        auto sizeAttr = P4HIR::IntAttr::get(infIntTy, llvm::APInt(32, defaultTableSize));
        rewriter.create<P4HIR::TableSizeOp>(callOp.getLoc(), infIntTy, sizeAttr, nullptr);
        return newTable;
    }
};

struct SynthesizeTablesPass : public P4::P4MLIR::impl::SynthesizeTablesBase<SynthesizeTablesPass> {
    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp moduleOp = getOperation();

        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        target.addLegalDialect<P4HIR::P4HIRDialect, BMv2IR::BMv2IRDialect,
                               P4CoreLib::P4CoreLibDialect>();
        target.addDynamicallyLegalOp<P4HIR::CallOp>([](P4HIR::CallOp callOp) {
            auto controlApplyParent = callOp->getParentOfType<P4HIR::ControlApplyOp>();
            return controlApplyParent == nullptr;
        });
        patterns.add<SynthTableFromActionCall>(&context);
        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
            signalPassFailure();
    }
};

}  // anonymous namespace
