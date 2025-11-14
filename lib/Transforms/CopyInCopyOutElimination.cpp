// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#include "mlir/Analysis/AliasAnalysis.h"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-copyincopyout-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_COPYINCOPYOUTELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct CopyInCopyOutEliminationPass
    : public P4::P4MLIR::impl::CopyInCopyOutEliminationBase<CopyInCopyOutEliminationPass> {
    CopyInCopyOutEliminationPass() = default;
    void runOnOperation() override;
};

template <typename... EffectTypes>
bool hasNoInterveningEffect(Operation *start, Operation *end, Value ref,
                            llvm::function_ref<bool(Value, Value)> mayAlias) {
    // A boolean representing whether an intervening operation could have impacted
    // `ref`.
    bool hasSideEffect = false;

    // Check whether the effect on ref can be caused by a given operation op.
    std::function<void(Operation *)> checkOperation = [&](Operation *op) {
        // If the effect has alreay been found, early exit,
        if (hasSideEffect) return;

        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
            SmallVector<MemoryEffects::EffectInstance, 1> effects;
            memEffect.getEffects(effects);

            bool opMayHaveEffect = false;
            for (auto effect : effects) {
                if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

                // If op causes EffectType on a potentially aliasing location for
                // memOp, mark as having the effect.
                if (isa<EffectTypes...>(effect.getEffect())) {
                    if (effect.getValue() && effect.getValue() != ref &&
                        !mayAlias(effect.getValue(), ref))
                        continue;
                    opMayHaveEffect = true;
                    break;
                }
            }

            if (!opMayHaveEffect) return;

            // We have an op with a memory effect and we cannot prove if it
            // intervenes.
            hasSideEffect = true;
            return;
        }

        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
            // Recurse into the regions for this op and check whether the internal
            // operations may have the side effect `EffectType` on memOp.
            for (Region &region : op->getRegions())
                for (Block &block : region)
                    for (Operation &op : block) checkOperation(&op);
            return;
        }

        // Otherwise, conservatively assume generic operations have the effect
        // on the operation
        hasSideEffect = true;
    };

    // Check all paths from ancestor op `parent` to the operation `to` for the
    // effect. It is known that `to` must be contained within `parent`.
    auto until = [&](Operation *parent, Operation *to) {
        // TODO check only the paths from `parent` to `to`.
        // Currently we fallback and check the entire parent op, rather than
        // just the paths from the parent path, stopping after reaching `to`.
        // This is conservatively correct, but could be made more aggressive.
        assert(parent->isAncestor(to));
        checkOperation(parent);
    };

    // Check for all paths from operation `from` to operation `untilOp` for the
    // given memory effect on the value `ref`.
    std::function<void(Operation *, Operation *)> recur = [&](Operation *from, Operation *untilOp) {
        assert(from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
               "Checking for side effect between two operations without a common "
               "ancestor");

        // If the operations are in different regions, recursively consider all
        // path from `from` to the parent of `to` and all paths from the parent
        // of `to` to `to`.
        if (from->getParentRegion() != untilOp->getParentRegion()) {
            recur(from, untilOp->getParentOp());
            until(untilOp->getParentOp(), untilOp);
            return;
        }

        // Now, assuming that `from` and `to` exist in the same region, perform
        // a CFG traversal to check all the relevant operations.

        // Additional blocks to consider.
        SmallVector<Block *, 2> todoBlocks;
        {
            // First consider the parent block of `from` an check all operations
            // after `from`.
            for (auto iter = std::next(from->getIterator()), end = from->getBlock()->end();
                 iter != end && &*iter != untilOp; ++iter) {
                checkOperation(&*iter);
            }

            // If the parent of `from` doesn't contain `to`, add the successors
            // to the list of blocks to check.
            if (untilOp->getBlock() != from->getBlock())
                for (Block *succ : from->getBlock()->getSuccessors()) todoBlocks.push_back(succ);
        }

        SmallPtrSet<Block *, 4> done;
        // Traverse the CFG until hitting `to`.
        while (!todoBlocks.empty()) {
            Block *blk = todoBlocks.pop_back_val();
            if (done.count(blk)) continue;
            done.insert(blk);
            for (auto &op : *blk) {
                if (&op == untilOp) break;
                checkOperation(&op);
                if (&op == blk->getTerminator())
                    for (Block *succ : blk->getSuccessors()) todoBlocks.push_back(succ);
            }
        }
    };

    recur(start, end);
    return !hasSideEffect;
}

class CopyOutElimination : public mlir::OpRewritePattern<P4HIR::VariableOp> {
 public:
    CopyOutElimination(MLIRContext *context, AliasAnalysis &aliasAnalysis)
        : OpRewritePattern(context), aliasAnalysis(aliasAnalysis) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp alias,
                                        mlir::PatternRewriter &rewriter) const override {
        auto *block = alias->getBlock();
        auto aliasUsers = llvm::to_vector(alias->getUsers());

        auto mayAlias = [&](Value val1, Value val2) -> bool {
            return !aliasAnalysis.alias(val1, val2).isNo();
        };

        // The variable should have only 2 uses:
        //   - The instruction that writes to it (function / extern / action call)
        //   - The read out
        // For now we assume that all uses are within the same BB. This could be
        // changed with dominance condition later on if necessary.
        if (llvm::size(aliasUsers) != 2 || alias->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(
                alias, "alias variable does not have out alias use pattern");

        llvm::sort(aliasUsers,
                   [&](mlir::Operation *a, mlir::Operation *b) { return a->isBeforeInBlock(b); });

        // Last user must be read which, in turn, must have a single use in the current block.
        auto writeAliasOp = dyn_cast<MemoryEffectOpInterface>(aliasUsers.front());
        auto readOp = dyn_cast<P4HIR::ReadOp>(aliasUsers.back());
        if (!writeAliasOp || !readOp || !readOp->hasOneUse() || readOp->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(alias, "invalid alias use");

        // Find the read destination
        auto writeAliaseeOp = dyn_cast<P4HIR::AssignOp>(*readOp->getUsers().begin());
        if (!writeAliaseeOp)
            return rewriter.notifyMatchFailure(alias, "invalid alias value assignment");

        // Ensure that writeOp really writes to alias
        auto aliasee = writeAliaseeOp.getRef();

        // Now we are having the following set of ops:
        //  %alias = p4hir.variable
        //  ...
        //  <writeOp> op1, ..., %alias, ... opN
        //  ...
        //  %alias.val = p4hir.read %alias
        //  ...
        //  p4hir.assign %alias.val, %aliasee
        //
        // We want to transform this into:
        //  <writeOp> op1, ..., %aliasee, ... opN
        // eliminating variable and copies
        //
        // In order to do this we need to ensure:
        //   - None of op1, ..., opN alias %aliasee and <writeOp> only writes to the value
        //   - There is no intervening write to or read from %aliasee between writeOp and assign
        //   - Note that %aliasee might be a field of struct, header or array, so we
        //     need to check for sub- and super-field writes

        // Check for aliasing & memory effects of <writeOp>
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        writeAliasOp.getEffects(effects);
        for (const auto &effect : effects) {
            // Skip non-default resources, these never affect / alias normal values
            if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

            // <writeOp> should only write to %alias, reading is disallowed as it will be
            // an uninitialized read
            if (effect.getValue() == alias) {
                if (!mlir::isa<MemoryEffects::Write>(effect.getEffect()))
                    return rewriter.notifyMatchFailure(alias, "unsupported alias value write op");
                continue;
            }

            if (mayAlias(aliasee, effect.getValue()))
                return rewriter.notifyMatchFailure(alias, [&](auto &diag) {
                    diag << aliasee << " may alias " << effect.getValue();
                });
        }

        // Check for intervening memory effects on %aliasee
        if (!hasNoInterveningEffect<MemoryEffects::Write, MemoryEffects::Read>(
                writeAliasOp, writeAliaseeOp, aliasee, mayAlias))
            return rewriter.notifyMatchFailure(alias, "intervening write to the value");

        // We should be good now:
        //  - Replace %alias with %aliasee
        //  - Kill read out of %alias
        //  - Kill write to %aliasee
        rewriter.replaceOp(alias, aliasee);
        rewriter.eraseOp(writeAliaseeOp);
        rewriter.eraseOp(readOp);

        return mlir::success();
    }

 private:
    AliasAnalysis &aliasAnalysis;
};

class CopyInOutElimination : public mlir::OpRewritePattern<P4HIR::VariableOp> {
 public:
    CopyInOutElimination(MLIRContext *context, AliasAnalysis &aliasAnalysis)
        : OpRewritePattern(context), aliasAnalysis(aliasAnalysis) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp alias,
                                        mlir::PatternRewriter &rewriter) const override {
        auto *block = alias->getBlock();
        auto aliasUsers = llvm::to_vector(alias->getUsers());

        auto mayAlias = [&](Value val1, Value val2) -> bool {
            return !aliasAnalysis.alias(val1, val2).isNo();
        };

        // The variable should have only 3 uses:
        //   - The read in
        //   - The instruction that writes to it (function / extern / action call)
        //   - The read out
        // For now we assume that all uses are within the same BB. This could be
        // changed with dominance condition later on if necessary.
        if (llvm::size(aliasUsers) != 3 || alias->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(
                alias, "alias variable does not have inout alias use pattern");

        llvm::sort(aliasUsers,
                   [&](mlir::Operation *a, mlir::Operation *b) { return a->isBeforeInBlock(b); });

        // Last user must be read which, in turn, must have a single use in the current block.
        auto readInOp = dyn_cast<P4HIR::AssignOp>(aliasUsers.front());
        auto writeAliasOp = dyn_cast<MemoryEffectOpInterface>(aliasUsers[1]);
        auto readOutOp = dyn_cast<P4HIR::ReadOp>(aliasUsers.back());
        if (!writeAliasOp || !readInOp || !readOutOp || !readOutOp->hasOneUse() ||
            readOutOp->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(alias, "invalid alias use");

        // Find the read out destination (aliasee)
        auto writeAliaseeOp = dyn_cast<P4HIR::AssignOp>(*readOutOp->getUsers().begin());
        if (!writeAliaseeOp)
            return rewriter.notifyMatchFailure(alias, "invalid alias value assignment");

        auto aliasee = writeAliaseeOp.getRef();

        // Ensure that read in originates from aliasee
        auto readAliaseeOp = readInOp.getValue().getDefiningOp<P4HIR::ReadOp>();
        if (!readAliaseeOp || !readAliaseeOp->hasOneUse() || readAliaseeOp.getRef() != aliasee)
            return rewriter.notifyMatchFailure(alias, "invalid aliasee value read");

        // Now we are having the following set of ops:
        //  %alias = p4hir.variable
        //  ...
        //  %aliasee.val = p4hir.read %aliasee
        //  ...
        //  p4hir.assign %aliasee.val, %alias
        //  ...
        //  <writeOp> op1, ..., %alias, ... opN
        //  ...
        //  %alias.val = p4hir.read %alias
        //  ...
        //  p4hir.assign %alias.val, %aliasee
        //
        // We want to transform this into:
        //  <writeOp> op1, ..., %aliasee, ... opN
        // eliminating variable and copies
        //
        // In order to do this we need to ensure:
        //   - None of op1, ..., opN alias %aliasee and <writeOp> both reads writes to the value
        //   - There is no intervening write to or read from %aliasee between read in and assign
        //     (this is a bit conservative, we can allow reads before <writeOp>).
        //   - Note that %alias might be a field of struct, header or array, so we
        //     need to check for sub- and super-field writes

        // Check for aliasing & memory effects of <writeOp>
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        writeAliasOp.getEffects(effects);
        for (const auto &effect : effects) {
            // Skip non-default resources, these never affect / alias normal values
            if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

            if (effect.getValue() == alias) {
                if (!mlir::isa<MemoryEffects::Write, MemoryEffects::Read>(effect.getEffect()))
                    return rewriter.notifyMatchFailure(alias, "unsupported alias value write op");
                continue;
            }

            if (mayAlias(aliasee, effect.getValue())) {
                return rewriter.notifyMatchFailure(alias, [&](auto &diag) {
                    diag << aliasee << " may alias " << effect.getValue();
                });
            }
        }

        // Check for intervening memory effects on %aliasee
        if (!hasNoInterveningEffect<MemoryEffects::Write, MemoryEffects::Read>(
                writeAliasOp, writeAliaseeOp, aliasee, mayAlias))
            return rewriter.notifyMatchFailure(
                alias, "intervening read from or write to the aliasee after write op");

        if (!hasNoInterveningEffect<MemoryEffects::Write>(readAliaseeOp, writeAliasOp, aliasee,
                                                          mayAlias))
            return rewriter.notifyMatchFailure(alias,
                                               "intervening read from the aliasee before write op");

        // We should be good now:
        //  - Replace %alias with %aliasee
        //  - Kill read out of %alias
        //  - Kill write to %aliasee
        rewriter.replaceOp(alias, aliasee);
        rewriter.eraseOp(writeAliaseeOp);
        rewriter.eraseOp(readInOp);
        rewriter.eraseOp(readOutOp);
        rewriter.eraseOp(readAliaseeOp);

        return mlir::success();
    }

 private:
    AliasAnalysis &aliasAnalysis;
};

void CopyInCopyOutEliminationPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<CopyOutElimination, CopyInOutElimination>(patterns.getContext(),
                                                           getAnalysis<AliasAnalysis>());

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createCopyInCopyOutEliminationPass() {
    return std::make_unique<CopyInCopyOutEliminationPass>();
}
