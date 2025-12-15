#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/PatternMatch.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::IRUtils {

// Inline `scopeOp`'s body to its parent.
void inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp);

// If `op` is an operation somewhere within `block`, check if we can split it in three parts:
// One with operations before `op`, one with `op` and one with operations after `op`.
// This function should be called to check if it's possible to use `splitBlockAt`.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split block in three parts: One with operations before `op`, one with `op` and one with
// operations after `op`. Any scopes surrounding `op` may be inlined to perform the split.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    const llvm::Twine &suffix);

// Helper class to replace one operation in a state with multiple new states.
// During init it creates two "pre" and "post" states in which the code before and after the split
// point is put. Then any number of additional states between pre and post can be created with the
// createXXXState functions. Once done the user should either call finalize to commit changes or
// cancel and any changes will be undone.
class SplitStateRewriter {
 public:
    SplitStateRewriter(mlir::RewriterBase &rewriter, mlir::Operation *op)
        : rewriter(rewriter), op(op), step(CREATED) {}
    SplitStateRewriter(SplitStateRewriter &&) = delete;
    SplitStateRewriter &operator=(SplitStateRewriter &&) = delete;
    SplitStateRewriter(const SplitStateRewriter &) = delete;
    SplitStateRewriter &operator=(const SplitStateRewriter &) = delete;
    ~SplitStateRewriter() {
        assert((step == CANCELED || step == FINALIZED) && "Incorrect state on destruction");
    }

    // Initialize and check if splitting is feasible.
    mlir::LogicalResult init();

    // Create new intermediate state that transitions to `transitionTo` and move `ops` in it.
    // If `ops` is non-null the terminator of the block is erased once moved.
    P4HIR::ParserStateOp createSubState(const llvm::Twine &suffix,
                                        P4HIR::ParserStateOp transitionTo = {},
                                        mlir::Block *ops = nullptr);

    // Same as createSubState but transition to the "post" state specifically.
    P4HIR::ParserStateOp createJoinSubState(const llvm::Twine &suffix, mlir::Block *ops = nullptr) {
        return createSubState(suffix, postState, ops);
    }

    // Undo any changes done so far.
    void cancel();

    // Finalize and commit all changes.
    void finalize();

    mlir::Operation *getSplitOp() { return op; }
    P4HIR::ParserStateOp getState() { return op->getParentOfType<P4HIR::ParserStateOp>(); }

    P4HIR::ParserStateOp getPreState() {
        assert(step == INITIALIZED);
        return preState;
    }

    P4HIR::ParserStateOp getPostState() {
        assert(step == INITIALIZED);
        return postState;
    }

    void setStateInsertionPointAfter(P4HIR::ParserStateOp afterState) {
        stateCreationPoint = afterState;
    }

 private:
    enum RewriteStep { CREATED, INITIALIZED, CANCELED, FINALIZED };

    mlir::RewriterBase &rewriter;
    mlir::Operation *op;
    RewriteStep step;

    P4HIR::ParserStateOp preState;
    P4HIR::ParserStateOp postState;
    mlir::Operation *stateCreationPoint;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
