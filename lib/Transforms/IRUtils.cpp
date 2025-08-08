#include "p4mlir/Transforms/IRUtils.h"

using namespace P4::P4MLIR;

void IRUtils::inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp) {
    mlir::Block *block = &scopeOp.getScopeRegion().front();
    mlir::Operation *terminator = block->getTerminator();
    mlir::ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, scopeOp, /*blockArgs=*/{});
    rewriter.replaceOp(scopeOp, results);
    rewriter.eraseOp(terminator);
}

bool IRUtils::canSplitBlockAt(mlir::Block *block, mlir::Operation *op) {
    assert(block->findAncestorOpInBlock(*op) != nullptr);

    while (true) {
        if (op->getBlock() == block) return true;
        op = op->getParentOp();
        if (!op || !mlir::isa<P4HIR::ScopeOp>(op)) return false;
    }
}

std::array<mlir::Block *, 3> IRUtils::splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                                   mlir::Operation *op) {
    assert(canSplitBlockAt(block, op));

    // Inline all scopes surrounding `op`.
    block->walk<mlir::WalkOrder::PostOrder>(
        [&](P4HIR::ScopeOp scopeOp) { inlineScope(rewriter, scopeOp); });

    // Split resulting block.
    mlir::Block *before = op->getBlock();
    mlir::Block *middle = rewriter.splitBlock(before, op->getIterator());
    mlir::Block *after = rewriter.splitBlock(middle, std::next(op->getIterator()));
    return {before, middle, after};
}

namespace {

using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;

// Returns blocks that contain uses of val and are not ancestors of `block`.
BlockSet getEscapingBlocks(mlir::Block *block, mlir::Value val) {
    BlockSet escapingBlocks;
    for (mlir::Operation *user : val.getUsers())
        if (user->getBlock() != block && !block->findAncestorOpInBlock(*user))
            escapingBlocks.insert(user->getBlock());
    return escapingBlocks;
}

// Potentially promote `val` to a local variable.
// The rewriter's insertion point is the location where the new variable is created.
void promoteValToVar(mlir::RewriterBase &rewriter, mlir::Value val, const BlockSet &escapingBlocks,
                     llvm::StringRef varName = "promoted_local") {
    if (escapingBlocks.empty()) return;

    // Create new variable to hold `val`.
    auto newVar = rewriter.create<P4HIR::VariableOp>(
        val.getLoc(), P4HIR::ReferenceType::get(val.getType()), varName);

    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Insert a new read in all escaping blocks and replace uses of `val` in that block.
    for (mlir::Block *block : escapingBlocks) {
        rewriter.setInsertionPointToStart(block);
        auto newVal = rewriter.create<P4HIR::ReadOp>(val.getLoc(), newVar);
        rewriter.replaceUsesWithIf(val, newVal, [block](mlir::OpOperand &use) {
            return use.getOwner()->getBlock() == block;
        });
    }

    // Assign the new variable after `val`'s definition.
    rewriter.setInsertionPointAfterValue(val);
    rewriter.create<P4HIR::AssignOp>(val.getLoc(), val, newVar);
}

// Replace uses of `op` with a copy in all escaping blocks.
void copyOp(mlir::RewriterBase &rewriter, mlir::Operation *op, const BlockSet &escapingBlocks) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    for (mlir::Block *block : escapingBlocks) {
        rewriter.setInsertionPointToStart(block);
        auto newOp = rewriter.clone(*op);
        rewriter.replaceOpUsesWithinBlock(op, newOp->getResults(), block);
    }
}

};  // namespace

// Corrects all values that are used outside `block` due to splitting.
// Rewriter position must be in a place where we can create locals (e.g. parser or control locals).
void IRUtils::adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block) {
    // Iterate in reverse so we process uses before their definition in this block.
    // This is needed to properly handle operands after `copyOp` calls.
    for (mlir::Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            // Move variables to top-level if needed.
            if (!getEscapingBlocks(block, op.getResult(0)).empty())
                rewriter.moveOpBefore(&op, rewriter.getInsertionBlock(),
                                      rewriter.getInsertionPoint());
        } else if (mlir::isa<P4HIR::ConstOp, P4HIR::StructExtractRefOp, P4HIR::ArrayElementRefOp,
                             P4HIR::SliceRefOp>(op)) {
            // Copy RefOps and constants.
            auto escapingBlocks = getEscapingBlocks(block, op.getResult(0));
            copyOp(rewriter, &op, escapingBlocks);
        } else {
            // Promote values to variables as needed.
            for (auto val : op.getResults())
                promoteValToVar(rewriter, val, getEscapingBlocks(block, val));
        }
    }
}

P4HIR::ParserStateOp IRUtils::createSubState(mlir::RewriterBase &rewriter,
                                             P4HIR::ParserStateOp state,
                                             const llvm::Twine &suffix) {
    auto parser = state->getParentOfType<P4HIR::ParserOp>();
    auto getUniqueName = [&](const llvm::Twine &basename) {
        unsigned counter = 0;
        return mlir::SymbolTable::generateSymbolName<256>(
            basename.str(),
            [&](llvm::StringRef candidate) { return parser.lookupSymbol(candidate) != nullptr; },
            counter);
    };

    auto name = getUniqueName(llvm::Twine(state.getSymName()) + "_" + suffix);
    auto newState =
        rewriter.create<P4HIR::ParserStateOp>(state.getLoc(), name, mlir::DictionaryAttr());
    rewriter.createBlock(&newState.getBody(), newState.getBody().begin());

    return newState;
};

mlir::LogicalResult IRUtils::SplitStateRewriter::init() {
    assert(step == CREATED || step == CANCELED);
    auto state = getState();

    if (!state)
        return rewriter.notifyMatchFailure(op->getLoc(), "Operation must be within parser state.");

    if (!canSplitBlockAt(state.getBlock(), op))
        return rewriter.notifyMatchFailure(op->getLoc(), "Cannot split state at given position.");

    step = INITIALIZED;

    // Create two new sub-states "pre" and "post" that will hold the code before and after `op`
    // respectively.
    setStateInsertionPointAfter(state);
    preState = createSubState("pre");
    postState = createSubState("post");
    setStateInsertionPointAfter(preState);

    return mlir::success();
}

P4HIR::ParserStateOp IRUtils::SplitStateRewriter::createSubState(const llvm::Twine &suffix,
                                                                 P4HIR::ParserStateOp transitionTo,
                                                                 mlir::Block *ops) {
    assert(step == INITIALIZED);
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPointAfter(stateCreationPoint);
    auto newState = IRUtils::createSubState(rewriter, getState(), suffix);
    auto newStateBB = newState.getBlock();

    if (ops) {
        rewriter.inlineBlockBefore(ops, newStateBB, newStateBB->end());
        rewriter.eraseOp(newStateBB->getTerminator());
    }

    if (transitionTo) {
        rewriter.setInsertionPointToEnd(newStateBB);
        rewriter.create<P4HIR::ParserTransitionOp>(newState.getLoc(), transitionTo.getSymbolRef());
    }

    stateCreationPoint = newState;

    return newState;
}

void IRUtils::SplitStateRewriter::cancel() {
    assert(step == CREATED || step == INITIALIZED);

    if (step == INITIALIZED) {
        auto createdStates = llvm::make_range(preState->getIterator(), ++postState->getIterator());
        for (mlir::Operation &op : llvm::make_early_inc_range(createdStates)) rewriter.eraseOp(&op);
    }

    step = CANCELED;
}

void IRUtils::SplitStateRewriter::finalize() {
    assert(step == INITIALIZED);

    auto state = getState();
    assert(canSplitBlockAt(state.getBlock(), op) && "Split op moved to invalid location");
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Split code around `op` and move it to "pre" / "post" states.
    auto [beforeBB, opBB, afterBB] = splitBlockAt(rewriter, state.getBlock(), op);

    auto preStateBB = preState.getBlock();
    rewriter.inlineBlockBefore(beforeBB, preStateBB, preStateBB->begin());
    auto postStateBB = postState.getBlock();
    rewriter.inlineBlockBefore(afterBB, postStateBB, postStateBB->begin());

    // Replace `op` with a transition to the "pre" state.
    rewriter.setInsertionPoint(op);
    rewriter.create<P4HIR::ParserTransitionOp>(op->getLoc(), preState.getSymbolRef());
    rewriter.eraseOp(op);

    // Due to the splitting states, we may have values with uses in other states.
    auto parser = state->getParentOfType<P4HIR::ParserOp>();
    rewriter.setInsertionPoint(parser.getStartState());
    adjustBlockUses(rewriter, preStateBB);

    step = FINALIZED;
}
