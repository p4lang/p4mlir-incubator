// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/P4HIR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
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

struct IfOpFlattening : public OpRewritePattern<P4HIR::IfOp> {
    using OpRewritePattern<P4HIR::IfOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::IfOp ifOp,
                                        mlir::PatternRewriter &rewriter) const override {
        // Start by splitting the block containing the if into two parts. The part before will
        // contain the condition, the part after will be the continuation point.
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto loc = ifOp.getLoc();
        auto *condBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
        mlir::Block *continueBlock;
        if (ifOp.getNumResults() == 0) {
            continueBlock = remainingOpsBlock;
        } else {
            continueBlock =
                rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes(),
                                     llvm::SmallVector<mlir::Location>(ifOp.getNumResults(), loc));
            P4HIR::BrOp::create(rewriter, loc, remainingOpsBlock);
        }

        // Move blocks from the "then" region to the region containing if, place it before the
        // continuation block, and branch to it.
        auto &thenRegion = ifOp.getThenRegion();
        auto *thenBlock = &thenRegion.front();
        mlir::Operation *thenTerminator = thenRegion.back().getTerminator();
        mlir::ValueRange thenTerminatorOperands = thenTerminator->getOperands();
        rewriter.setInsertionPointToEnd(&thenRegion.back());
        P4HIR::BrOp::create(rewriter, loc, continueBlock, thenTerminatorOperands);
        rewriter.eraseOp(thenTerminator);
        rewriter.inlineRegionBefore(thenRegion, continueBlock);

        // Move blocks from the "else" region (if present) to the region containing if, place it
        // before the continuation block and branch to it. It will be placed after the "then"
        // regions.
        auto *elseBlock = continueBlock;
        auto &elseRegion = ifOp.getElseRegion();
        if (!elseRegion.empty()) {
            elseBlock = &elseRegion.front();
            mlir::Operation *elseTerminator = elseRegion.back().getTerminator();
            mlir::ValueRange elseTerminatorOperands = elseTerminator->getOperands();
            rewriter.setInsertionPointToEnd(&elseRegion.back());
            P4HIR::BrOp::create(rewriter, loc, continueBlock, elseTerminatorOperands);
            rewriter.eraseOp(elseTerminator);
            rewriter.inlineRegionBefore(elseRegion, continueBlock);
        }

        rewriter.setInsertionPointToEnd(condBlock);
        P4HIR::CondBrOp::create(rewriter, loc, ifOp.getCondition(), thenBlock, elseBlock);

        // Ok, we're done!
        rewriter.replaceOp(ifOp, continueBlock->getArguments());
        return mlir::success();
    }
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
        P4HIR::BrOp::create(rewriter, loc, mlir::ValueRange(), beforeBody);

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

struct ForOpFlattening : public OpRewritePattern<P4HIR::ForOp> {
    using OpRewritePattern<P4HIR::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::ForOp forOp, PatternRewriter &rewriter) const override {
        auto loc = forOp.getLoc();

        // Split the block at the for
        auto *condBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *exitBlock = rewriter.splitBlock(condBlock, opPosition);

        Region &condRegion = forOp.getCondRegion();
        Block *condEntry = &condRegion.front();
        Block *condExit = &condRegion.back();
        Region &bodyRegion = forOp.getBodyRegion();
        Block *bodyEntry = &bodyRegion.front();
        Block *bodyExit = &bodyRegion.back();
        Region &updatesRegion = forOp.getUpdatesRegion();
        Block *updatesEntry = &updatesRegion.front();
        Block *updatesExit = &updatesRegion.back();

        // p4hir.condition -> cond_br body, exit
        auto conditionOp = cast<P4HIR::ConditionOp>(condExit->getTerminator());
        rewriter.setInsertionPointToEnd(condExit);
        P4HIR::CondBrOp::create(rewriter, loc, conditionOp.getCondition(), bodyEntry, exitBlock);
        rewriter.eraseOp(conditionOp);

        // body, p4hir.yield -> br updates
        Operation *bodyYield = bodyExit->getTerminator();
        rewriter.setInsertionPointToEnd(bodyExit);
        P4HIR::BrOp::create(rewriter, loc, updatesEntry);
        rewriter.eraseOp(bodyYield);

        // updates backedge
        Operation *updatesYield = updatesExit->getTerminator();
        rewriter.setInsertionPointToEnd(updatesExit);
        P4HIR::BrOp::create(rewriter, loc, condEntry);
        rewriter.eraseOp(updatesYield);

        rewriter.inlineRegionBefore(condRegion, exitBlock);
        rewriter.inlineRegionBefore(bodyRegion, exitBlock);
        rewriter.inlineRegionBefore(updatesRegion, exitBlock);

        rewriter.setInsertionPointToEnd(condBlock);
        P4HIR::BrOp::create(rewriter, loc, condEntry);

        rewriter.eraseOp(forOp);
        return success();
    }
};

// Lowers `p4hir.foreach` to a do-while loop.
// For `range`s, the induction variable iterates the
// inclusive range [lo, hi]:
//     cond_br (lo <= hi) ^body(lo), ^exit
//   ^body(%i):
//     ...
//     cond_br (%i == hi) ^exit, ^inc
//   ^inc:
//     br ^body(%i + 1)
//   ^exit:
// `p4hir.foreach` over Arrays and Header Stacks are lowered similarly,
// with just an extra block that retrieves the actual element from the collection.
struct ForInFlattening : public OpRewritePattern<P4HIR::ForInOp> {
    using OpRewritePattern<P4HIR::ForInOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::ForInOp forInOp,
                                  PatternRewriter &rewriter) const override {
        mlir::Value collection = forInOp.getCollection();
        auto loc = forInOp.getLoc();

        rewriter.setInsertionPoint(forInOp);
        mlir::Value lo;
        mlir::Value hi;
        // Array indexed to get the element bound in the loop body. Unset for ranges, where the
        // index is the loop variable itself.
        mlir::Value array;

        mlir::OpFoldResult loBound;
        mlir::OpFoldResult hiBound;
        if (matchPattern(collection, m_Range(&loBound, &hiBound))) {
            // Materialize loop bounds
            auto materialize = [&](mlir::OpFoldResult bound) -> mlir::Value {
                if (auto value = mlir::dyn_cast<mlir::Value>(bound)) return value;
                return P4HIR::ConstOp::create(
                    rewriter, loc, mlir::cast<mlir::TypedAttr>(mlir::cast<mlir::Attribute>(bound)));
            };
            lo = materialize(loBound);
            hi = materialize(hiBound);
        } else {
            size_t size;
            auto hsType = mlir::dyn_cast<P4HIR::HeaderStackType>(collection.getType());
            if (auto arrType = mlir::dyn_cast<P4HIR::ArrayType>(collection.getType()))
                size = arrType.getSize();
            else if (hsType)
                size = hsType.getArraySize();
            else
                return failure();

            // Empty collection: the body is never executed.
            if (size == 0) {
                rewriter.eraseOp(forInOp);
                return success();
            }

            // Header stack data is loop invariant, extract it once before the loop.
            array = hsType ? P4HIR::StructExtractOp::create(rewriter, loc, collection,
                                                            P4HIR::HeaderStackType::dataFieldName)
                           : collection;

            unsigned width = std::max(1U, llvm::Log2_64_Ceil(static_cast<uint64_t>(size)));
            auto indexType = P4HIR::BitsType::get(rewriter.getContext(), width, false);
            lo = P4HIR::ConstOp::create(rewriter, loc, P4HIR::IntAttr::get(indexType, 0));
            hi = P4HIR::ConstOp::create(rewriter, loc, P4HIR::IntAttr::get(indexType, size - 1));
        }

        // Split the block at the foreach
        auto *entryBlock = forInOp->getBlock();
        auto *exitBlock = rewriter.splitBlock(entryBlock, forInOp->getIterator());

        Region &bodyRegion = forInOp.getBodyRegion();
        Block *bodyEntry = &bodyRegion.front();
        Block *bodyExit = &bodyRegion.back();

        // For collections, prepend a block taking the index and reading the element.
        if (array) {
            Block *header = rewriter.createBlock(bodyEntry, {lo.getType()}, {loc});
            auto elem = P4HIR::ArrayGetOp::create(rewriter, loc, array, header->getArgument(0));
            if (bodyExit == bodyEntry) bodyExit = header;
            rewriter.mergeBlocks(bodyEntry, header, mlir::ValueRange{elem});
            bodyEntry = header;
        }
        mlir::Value i = bodyEntry->getArgument(0);

        // increment: br ^body(%i + 1)
        auto *incBlock = rewriter.createBlock(exitBlock);
        auto one = P4HIR::ConstOp::create(rewriter, loc, P4HIR::IntAttr::get(i.getType(), 1));
        auto next = P4HIR::BinOp::create(rewriter, loc, P4HIR::BinOpKind::Add, i, one);
        P4HIR::BrOp::create(rewriter, loc, bodyEntry, mlir::ValueRange{next});

        // body: ... cond_br (%i == hi) ^exit, ^inc
        Operation *bodyYield = bodyExit->getTerminator();
        rewriter.setInsertionPoint(bodyYield);
        auto isLast = P4HIR::CmpOp::create(rewriter, loc, P4HIR::CmpOpKind::Eq, i, hi);
        P4HIR::CondBrOp::create(rewriter, loc, isLast, exitBlock, incBlock);
        rewriter.eraseOp(bodyYield);

        rewriter.inlineRegionBefore(bodyRegion, incBlock);

        // Enter the loop, checking for empty ranges
        rewriter.setInsertionPointToEnd(entryBlock);
        if (!array) {
            auto nonEmpty = P4HIR::CmpOp::create(rewriter, loc, P4HIR::CmpOpKind::Le, lo, hi);
            P4HIR::CondBrOp::create(rewriter, loc, nonEmpty, bodyEntry, exitBlock,
                                    mlir::ValueRange{lo});
        } else {
            // We don't emit the loop at all if the collection is empty
            P4HIR::BrOp::create(rewriter, loc, bodyEntry, mlir::ValueRange{lo});
        }

        rewriter.eraseOp(forInOp);
        return success();
    }
};

void FlattenCFGPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());

    patterns.add<IfOpFlattening, ScopeOpFlattening, ForOpFlattening, ForInFlattening>(
        patterns.getContext());

    // Collect operations to apply patterns.
    llvm::SmallVector<Operation *, 16> ops;
    getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
        if (mlir::isa<P4HIR::IfOp, P4HIR::ScopeOp, P4HIR::ForOp, P4HIR::ForInOp>(op))
            ops.push_back(op);
    });

    // Apply patterns.
    if (applyOpPatternsGreedily(ops, std::move(patterns)).failed()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createFlattenCFGPass() {
    return std::make_unique<FlattenCFGPass>();
}
