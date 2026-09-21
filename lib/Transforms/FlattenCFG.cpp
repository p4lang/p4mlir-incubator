// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <functional>

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

// Lowers `p4hir.foreach` into a `p4hir.for`, depending on the collection:
//   - range:        counter is the element itself, iterating [lo, hi] (cmp le);
//   - array/stack:  counter is an index, iterating [0, size) (cmp lt), with the
//                   element read out of the collection at that index.
struct ForInLowering : public OpRewritePattern<P4HIR::ForInOp> {
    using OpRewritePattern<P4HIR::ForInOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::ForInOp forInOp,
                                  PatternRewriter &rewriter) const override {
        mlir::Value collection = forInOp.getCollection();
        auto loc = forInOp.getLoc();
        auto *body = &forInOp.getBodyRegion().front();

        mlir::Type counterType;
        P4HIR::CmpOpKind cmpKind;
        // Materialize loop bounds;
        std::function<mlir::Value(mlir::OpBuilder &, mlir::Location)> makeLow;
        std::function<mlir::Value(mlir::OpBuilder &, mlir::Location)> makeHigh;
        // Maps the current counter value to the element bound in the loop body.
        std::function<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::Value)> getElement;

        auto rangeOp = collection.getDefiningOp<P4HIR::RangeOp>();
        mlir::TypedAttr loAttr;
        mlir::TypedAttr hiAttr;
        bool isRangeSet = !rangeOp && matchPattern(collection, m_RangeSet(&loAttr, &hiAttr));
        if (rangeOp || isRangeSet) {
            // foreach over an inclusive [lo, hi] range: the counter is the element.
            cmpKind = P4HIR::CmpOpKind::Le;
            getElement = [](mlir::OpBuilder &, mlir::Location, mlir::Value i) { return i; };
            if (rangeOp) {
                auto lo = rangeOp.getLhs();
                auto hi = rangeOp.getRhs();
                counterType = lo.getType();
                makeLow = [lo](mlir::OpBuilder &, mlir::Location) { return lo; };
                makeHigh = [hi](mlir::OpBuilder &, mlir::Location) { return hi; };
            } else {
                // Constant range
                counterType = loAttr.getType();
                makeLow = [loAttr](mlir::OpBuilder &b, mlir::Location l) {
                    return P4HIR::ConstOp::create(b, l, loAttr);
                };
                makeHigh = [hiAttr](mlir::OpBuilder &b, mlir::Location l) {
                    return P4HIR::ConstOp::create(b, l, hiAttr);
                };
            }
        } else {
            size_t size;
            if (auto arrType = mlir::dyn_cast<P4HIR::ArrayType>(collection.getType())) {
                size = arrType.getSize();
                getElement = [collection](mlir::OpBuilder &b, mlir::Location l, mlir::Value i) {
                    return P4HIR::ArrayGetOp::create(b, l, collection, i);
                };
            } else if (auto hsType = mlir::dyn_cast<P4HIR::HeaderStackType>(collection.getType())) {
                size = hsType.getArraySize();
                getElement = [collection](mlir::OpBuilder &b, mlir::Location l, mlir::Value i) {
                    auto data = P4HIR::StructExtractOp::create(
                        b, l, collection, P4HIR::HeaderStackType::dataFieldName);
                    return P4HIR::ArrayGetOp::create(b, l, data, i);
                };
            } else {
                return failure();
            }

            unsigned width = std::max(1U, llvm::Log2_64_Ceil(static_cast<uint64_t>(size) + 1));
            counterType = P4HIR::BitsType::get(rewriter.getContext(), width, false);
            cmpKind = P4HIR::CmpOpKind::Lt;
            makeLow = [counterType](mlir::OpBuilder &b, mlir::Location l) {
                return P4HIR::ConstOp::create(b, l, P4HIR::IntAttr::get(counterType, 0));
            };
            makeHigh = [counterType, size](mlir::OpBuilder &b, mlir::Location l) {
                return P4HIR::ConstOp::create(b, l, P4HIR::IntAttr::get(counterType, size));
            };
        }

        rewriter.setInsertionPoint(forInOp);
        auto iv = P4HIR::VariableOp::create(rewriter, loc, P4HIR::ReferenceType::get(counterType),
                                            "i", true);
        P4HIR::AssignOp::create(rewriter, loc, makeLow(rewriter, loc), iv);

        P4HIR::ForOp::create(
            rewriter, loc, forInOp.getAnnotations().value_or(nullptr),
            [&](mlir::OpBuilder &b, mlir::Location l) {
                auto i = P4HIR::ReadOp::create(b, l, counterType, iv);
                auto cond = P4HIR::CmpOp::create(b, l, cmpKind, i, makeHigh(b, l));
                P4HIR::ConditionOp::create(b, l, cond);
            },
            [&](mlir::OpBuilder &b, mlir::Location l) {
                auto i = P4HIR::ReadOp::create(b, l, counterType, iv);
                auto elem = getElement(b, l, i);
                rewriter.mergeBlocks(body, b.getInsertionBlock(), mlir::ValueRange{elem});
            },
            [&](mlir::OpBuilder &b, mlir::Location l) {
                auto i = P4HIR::ReadOp::create(b, l, counterType, iv);
                auto one = P4HIR::ConstOp::create(b, l, P4HIR::IntAttr::get(counterType, 1));
                auto next = P4HIR::BinOp::create(b, l, P4HIR::BinOpKind::Add, i, one);
                P4HIR::AssignOp::create(b, l, next, iv);
                P4HIR::YieldOp::create(b, l);
            });

        rewriter.eraseOp(forInOp);
        return success();
    }
};

void FlattenCFGPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());

    patterns.add<IfOpFlattening, ScopeOpFlattening, ForOpFlattening, ForInLowering>(
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
