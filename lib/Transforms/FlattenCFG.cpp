#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-flatten-cfg"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_FLATTENCFG
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct FlattenCFGPass : public P4::P4MLIR::impl::FlattenCFGBase<FlattenCFGPass> {
    FlattenCFGPass() = default;
    void runOnOperation() override;
};

class ScopeOpFlattening : public mlir::OpRewritePattern<P4HIR::ScopeOp> {
 public:
    using OpRewritePattern<P4HIR::ScopeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ScopeOp scopeOp,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto loc = scopeOp.getLoc();

        // Empty scope: just remove it.
        // TODO: Decide if we'd need to do something with annotated scopes
        if (scopeOp.isEmpty()) {
            rewriter.eraseOp(scopeOp);
            return mlir::success();
        }

        // Split the current block before the ScopeOp to create the inlining point.
        auto *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *afterBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        if (scopeOp.getNumResults() > 0) afterBlock->addArguments(scopeOp.getResultTypes(), loc);

        // Inline body region.
        auto *beforeBody = &scopeOp.getScopeRegion().front();
        auto *afterBody = &scopeOp.getScopeRegion().back();
        rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), afterBlock);

        // Save stack and then branch into the body of the region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<P4HIR::BrOp>(loc, mlir::ValueRange(), beforeBody);

        // Replace the scope return with a branch that jumps out of the body.
        rewriter.setInsertionPointToEnd(afterBody);
        if (auto yieldOp = dyn_cast<P4HIR::YieldOp>(afterBody->getTerminator())) {
            rewriter.replaceOpWithNewOp<P4HIR::BrOp>(yieldOp, yieldOp.getArgs(), afterBlock);
        }

        // Replace the op with values return from the body region.
        rewriter.replaceOp(scopeOp, afterBlock->getArguments());

        return mlir::success();
    }
};

void FlattenCFGPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());

    patterns.add<ScopeOpFlattening>(patterns.getContext());

    // Collect operations to apply patterns.
    llvm::SmallVector<Operation *, 16> ops;
    getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
        if (isa<P4HIR::ScopeOp>(op)) ops.push_back(op);
    });

    // Apply patterns.
    if (applyOpPatternsGreedily(ops, std::move(patterns)).failed()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createFlattenCFGPass() {
    return std::make_unique<FlattenCFGPass>();
}
