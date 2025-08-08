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

// Return true if it's valid to call `splitBlockAt` for the given arguments.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split `block` that has as ancestor `op` in three:
// One with operations before `op`, one with `op` and one with operations after `op`.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    llvm::StringRef suffix);

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
    P4HIR::ParserStateOp createSubState(llvm::StringRef suffix,
                                        P4HIR::ParserStateOp transitionTo = {},
                                        mlir::Block *ops = nullptr);

    // Same as createSubState but transition to the "post" state specifically.
    P4HIR::ParserStateOp createJoinSubState(llvm::StringRef suffix, mlir::Block *ops = nullptr) {
        return createSubState(suffix, postState, ops);
    }

    // Undo any changes done so far.
    void cancel();

    // Finalize and commit all changes.
    void finalize();

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

    P4HIR::ParserStateOp state;
    P4HIR::ParserStateOp preState;
    P4HIR::ParserStateOp postState;
    mlir::Operation *stateCreationPoint;
};

// Helper class to build transition select statements.
class TransitionSelectBuilder {
 public:
    TransitionSelectBuilder(mlir::RewriterBase &rewriter) : rewriter(rewriter) {}

    // Create a new transition select statement.
    void create(mlir::Location loc, mlir::Value selectArg) {
        assert((selectBody == nullptr) && "create can only be called once");
        selectOp = rewriter.create<P4HIR::ParserTransitionSelectOp>(loc, selectArg);
        selectBody = &selectOp.getBody().emplaceBlock();
    }

    // Add a new case that transitions to `transitionTo` using a custom yield builder.
    P4HIR::ParserSelectCaseOp addCase(
        llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> yieldBuilder,
        P4HIR::ParserStateOp transitionTo);

    // Add a new case that transitions to `transitionTo` with a constant keyset.
    P4HIR::ParserSelectCaseOp addCase(mlir::TypedAttr constant, P4HIR::ParserStateOp transitionTo);

    // Add a new case that transitions to `transitionTo` with a constant boolean keyset.
    P4HIR::ParserSelectCaseOp addCase(bool labelValue, P4HIR::ParserStateOp transitionTo) {
        return addCase(P4HIR::BoolAttr::get(rewriter.getContext(), labelValue), transitionTo);
    }

    // Add a new case that transitions to `transitionTo` with a keyset equivalent to `caseOp`.
    P4HIR::ParserSelectCaseOp addCase(P4HIR::CaseOp caseOp, P4HIR::ParserStateOp transitionTo);

    // Add a default case that transitions to `transitionTo`.
    P4HIR::ParserSelectCaseOp addDefaultCase(P4HIR::ParserStateOp transitionTo) {
        return addCase(P4HIR::UniversalSetAttr::get(rewriter.getContext()), transitionTo);
    }

 private:
    mlir::RewriterBase &rewriter;
    mlir::Block *selectBody = nullptr;
    P4HIR::ParserTransitionSelectOp selectOp;
    P4HIR::ParserSelectCaseOp lastAdded;
};

};  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
