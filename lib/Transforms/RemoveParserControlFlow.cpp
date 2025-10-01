#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-parser-control-flow"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVEPARSERCONTROLFLOW
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveParserControlFlowPass
    : public P4::P4MLIR::impl::RemoveParserControlFlowBase<RemoveParserControlFlowPass> {
    RemoveParserControlFlowPass() = default;
    void runOnOperation() override;
};

struct IfOpLowering : public mlir::OpRewritePattern<P4HIR::IfOp> {
    using OpRewritePattern<P4HIR::IfOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::IfOp ifOp,
                                        mlir::PatternRewriter &rewriter) const override {
        IRUtils::SplitStateRewriter ssr(rewriter, ifOp);
        IRUtils::TransitionSelectBuilder tsb(rewriter);

        if (ssr.init().failed()) return mlir::failure();

        // Create the corresponding select statement.
        rewriter.setInsertionPointToEnd(ssr.getPreState().getBlock());
        tsb.create(ifOp.getLoc(), ifOp.getCondition());

        auto thenState = ssr.createJoinSubState("then", &ifOp.getThenRegion().front());
        tsb.addCase(true, thenState);

        // If the ifOp has an else block, then similarly create an "else" state for it.
        P4HIR::ParserStateOp elseState;
        if (!ifOp.getElseRegion().empty()) {
            auto elseState = ssr.createJoinSubState("else", &ifOp.getElseRegion().front());
            tsb.addDefaultCase(elseState);
        } else {
            // Otherwise the else state is the "post" state.
            tsb.addDefaultCase(ssr.getPostState());
        }

        ssr.finalize();

        return mlir::success();
    }
};

struct SwitchOpLowering : public mlir::OpRewritePattern<P4HIR::SwitchOp> {
    using OpRewritePattern<P4HIR::SwitchOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::SwitchOp switchOp,
                                        mlir::PatternRewriter &rewriter) const override {
        IRUtils::SplitStateRewriter ssr(rewriter, switchOp);
        IRUtils::TransitionSelectBuilder tsb(rewriter);

        if (ssr.init().failed()) return mlir::failure();

        // Create the corresponding select statement.
        rewriter.setInsertionPointToEnd(ssr.getPreState().getBlock());
        tsb.create(switchOp.getLoc(), switchOp.getCondition());

        // Create transitions for each case.
        for (auto [idx, caseOp] : llvm::enumerate(switchOp.cases())) {
            std::string name = (llvm::Twine("case") + std::to_string(idx)).str();
            auto state = ssr.createJoinSubState(name, caseOp.getBlock());
            tsb.addCases(caseOp, state);
        }

        ssr.finalize();

        return mlir::success();
    }
};

void RemoveParserControlFlowPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<IfOpLowering, SwitchOpLowering>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    mlir::GreedyRewriteConfig grc;
    grc.setUseTopDownTraversal(true);

    auto walkResult = getOperation()->walk([&](P4HIR::ParserOp parserOp) -> mlir::WalkResult {
        // Interrupt on failure.
        return applyPatternsGreedily(parserOp, frozenPatterns, grc);
    });

    if (walkResult.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::Pass> P4::P4MLIR::createRemoveParserControlFlowPass() {
    return std::make_unique<RemoveParserControlFlowPass>();
}
