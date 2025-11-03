#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-soft-cf"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVESOFTCF
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveSoftCFPass : public P4::P4MLIR::impl::RemoveSoftCFBase<RemoveSoftCFPass> {
    RemoveSoftCFPass() = default;
    void runOnOperation() override;
};

// Helper struct to create and update boolean guard variables for conditionally executing code.
struct GuardVariable {
    // Create and initialize to true the guard variable.
    void init(mlir::OpBuilder &b, mlir::Location loc, llvm::StringRef name) {
        guardVar = b.create<P4HIR::VariableOp>(
            loc, P4HIR::ReferenceType::get(P4HIR::BoolType::get(b.getContext())), name, true);
        assign(b, loc, true);
    }

    // Assign `value` to the guard variable.
    void assign(mlir::OpBuilder &b, mlir::Location loc, bool value) {
        assert(guardVar && "Guard not initialized");
        auto constTrue = b.create<P4HIR::ConstOp>(loc, P4HIR::BoolAttr::get(b.getContext(), value));
        b.create<P4HIR::AssignOp>(loc, constTrue, guardVar);
    }

    // Create and return a new block that is executed only when this guard is true.
    mlir::Block *createGuardedBlock(mlir::OpBuilder &b, mlir::Location loc) {
        assert(guardVar && "Guard not initialized");
        auto cond = b.create<P4HIR::ReadOp>(loc, guardVar);
        auto ifOp = b.create<P4HIR::IfOp>(
            loc, cond, false,
            [&](mlir::OpBuilder &b, mlir::Location) { P4HIR::buildTerminatedBody(b, loc); });
        return &ifOp.getThenRegion().front();
    }

    // Guard an existing block.
    void guardBlock(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Block *block) {
        assert(block->mightHaveTerminator());
        rewriter.setInsertionPointToStart(block);
        mlir::Operation *firstOp = &block->front();
        mlir::Block *guardedBlock = createGuardedBlock(rewriter, loc);
        auto opsToMove =
            llvm::make_range(firstOp->getIterator(), block->getTerminator()->getIterator());
        for (mlir::Operation &op : llvm::make_early_inc_range(opsToMove))
            rewriter.moveOpBefore(&op, guardedBlock->getTerminator());
    }

    P4HIR::VariableOp getVar() const { return guardVar; }
    operator bool() const { return static_cast<bool>(guardVar); }

 private:
    P4HIR::VariableOp guardVar;
};

// Helper struct to replace all soft control operations within a function-like operation.
// This is done by moving operations or introducing guard variables that control execution of
// subsequent operations. A core observation used is that if we're currently adjusting an operation
// at some "nest level" X, we can place all subsequent operations at nest level Y >= X while
// guaranteeing the original semantics are preserved (we never need to introduce a guarded block at
// an outer level). Operations are either moved to a more nested block or to a guard block that is
// introduced at the same level.
struct RemoveSoftCF {
    RemoveSoftCF(mlir::RewriterBase &rewriter) : rewriter(rewriter) {}

    void transform(mlir::Location loc, mlir::Region *region, mlir::Type returnType) {
        assert(region->hasOneBlock() && "Expected region with one block");
        mlir::Block *block = &region->front();

        rewriter.setInsertionPointToStart(block);
        returnGuard.init(rewriter, loc, "return_guard");

        // Create a variable to store the final return if needed.
        if (returnType && !mlir::isa<P4HIR::VoidType>(returnType))
            returnVar = rewriter.create<P4HIR::VariableOp>(
                loc, P4HIR::ReferenceType::get(returnType), "return_value", true);

        // Replace soft control flow in the function's body.
        visitBlock(block);

        if (returnVar) {
            // Update the real return statement.
            auto returnOp = mlir::cast<P4HIR::ReturnOp>(block->getTerminator());
            rewriter.setInsertionPoint(returnOp);
            auto returnValue = rewriter.create<P4HIR::ReadOp>(returnOp.getLoc(), returnVar);
            rewriter.modifyOpInPlace(returnOp, [&]() {
                returnOp.getInputMutable().assign(mlir::ValueRange(returnValue));
            });
        }
    }

 private:
    // Supported soft control flow operations.
    enum ControlFlowType : unsigned {
        CF_None = 0U,
        CF_Return = 1U << 0U,
        CF_Break = 1U << 1U,
        CF_Continue = 1U << 2U
    };

    // Enum to describe the possible places that execution continues after the execution of an
    // operation or block:
    //   - None: Any code following is unreachable.
    //   - Next: Execution continues normally to the next operations.
    //   - Nested: Execution continues in single nested point.
    //   - Multiple: Execution continues at multiple potential places.
    enum ExecutionType { ET_None, ET_Next, ET_Nested, ET_Multiple };

    // Struct holding control flow information for an operation or block.
    struct CFInfo {
        CFInfo() : cfTypes(CF_None), execType(ET_Next), execPoint(nullptr) {}

        // Constructor for soft control flow operations.
        CFInfo(ControlFlowType cfType, mlir::Operation *execPoint = nullptr)
            : cfTypes(cfType), execType(ET_None), execPoint(execPoint) {}

        // Bit mask of soft control flow types in this operation or block.
        unsigned cfTypes;
        // Describes how execution continues after this operation or block is executed.
        ExecutionType execType;
        // If an operation: holds the nested insertion point when `cp` is ET_Nested.
        // If a block: holds the place that execution would continue after execution of the block.
        mlir::Operation *execPoint;
    };

    // Computes control flow information for all operations in a block.
    CFInfo visitBlock(mlir::Block *block) {
        CFInfo blockInfo;
        mlir::Operation *execPoint = &block->front();

        for (mlir::Operation &op : llvm::make_early_inc_range(*block)) {
            if (block->mightHaveTerminator() && &op == block->getTerminator()) break;

            // Rest of the block is unreachable, erase ops.
            if (blockInfo.execType == ET_None) {
                SmallVector<Operation *> restOps;
                for (mlir::Operation &op :
                     llvm::make_range(op.getIterator(), block->getTerminator()->getIterator()))
                    restOps.push_back(&op);
                for (mlir::Operation *op : llvm::make_early_inc_range(llvm::reverse(restOps)))
                    rewriter.eraseOp(op);
                execPoint = nullptr;
                break;
            }

            // Materialize guards on demand.
            // Create execution guard and convert ET_Multiple to ET_Nested.
            if (blockInfo.execType == ET_Multiple) {
                blockInfo.execType = ET_Nested;
                rewriter.setInsertionPoint(execPoint);
                GuardVariable &currentGuard = isInLoop() ? continueGuard : returnGuard;
                mlir::Block *guardedBlock = currentGuard.createGuardedBlock(rewriter, op.getLoc());
                execPoint = &guardedBlock->front();
            }

            if (blockInfo.execType == ET_Nested)
                rewriter.moveOpBefore(&op, execPoint);
            else if (blockInfo.execType == ET_Next)
                execPoint = execPoint->getNextNode();

            auto opInfo = visitOp(block, &op);

            if (opInfo.execType != ET_Next) {
                blockInfo.cfTypes |= opInfo.cfTypes;
                blockInfo.execType = opInfo.execType;
                if (opInfo.execType == ET_Nested) execPoint = opInfo.execPoint;
            }
        }

        blockInfo.execPoint = execPoint;

        return blockInfo;
    }

    // Transform a conditional execution operation.
    // It must hold that exactly one of `blocks` is executed each time.
    CFInfo visitConditional(SmallVectorImpl<mlir::Block *> &blocks) {
        // The "Continue Point" for the entire conditional statement is:
        //  - None, if all blocks are None.
        //  - Nested, if a single block is Next/Nested and all others are None.
        //  - Multiple, in all other cases.
        CFInfo condInfo;
        condInfo.execType = ET_None;
        for (mlir::Block *block : blocks) {
            CFInfo caseInfo = visitBlock(block);
            condInfo.cfTypes |= caseInfo.cfTypes;

            if ((condInfo.execType == ET_None) &&
                (caseInfo.execType == ET_Next || caseInfo.execType == ET_Nested)) {
                condInfo.execPoint = caseInfo.execPoint;
                condInfo.execType = ET_Nested;
            } else if (caseInfo.execType != ET_None) {
                condInfo.execType = ET_Multiple;
            }
        }

        return condInfo;
    }

    bool isInLoop() const { return static_cast<bool>(breakGuard); }

    // Transform a loop operation.
    // `breakGuardBuilder` should transform `loop` so that the given break guard is used.
    // Apart from `returnGuard`, two additional guards are introduced per loop: `breakGuard` and
    // `continueGuard`. When `continueGuard` is set to false the rest of the current loop iteration
    // is skipped. When `breakGuard` is set to false the loop must not execute further than the
    // current iteration. All three guards are combined to implement loop control flow:
    //   - `return`   -> `returnGuard` + `breakGuard` + `continueGuard`
    //   - `break`    -> `breakGuard` + `continueGuard`
    //   - `continue` -> `continueGuard`
    // If a loop doesn't use all control flow kinds, then guards are merged for efficiency.
    CFInfo visitLoop(mlir::Operation *loop, mlir::Block *loopBody,
                     llvm::function_ref<void(GuardVariable &)> breakGuardBuilder) {
        auto oldBreakGuard = breakGuard;
        auto oldContinueGuard = continueGuard;
        breakGuard = {};
        continueGuard = {};

        // Create one break and one continue guard for the loop in advance.
        // If no break or continue statement affects this loop, the canonicalizer will remove the
        // variable and dead writes.
        rewriter.setInsertionPoint(loop);
        breakGuard.init(rewriter, loop->getLoc(), "loop_break_guard");
        continueGuard.init(rewriter, loop->getLoc(), "loop_continue_guard");

        // Reset `continueGuard` at start of each loop iteration.
        rewriter.setInsertionPointToStart(loopBody);
        continueGuard.assign(rewriter, loop->getLoc(), true);

        CFInfo bodyInfo = visitBlock(loopBody);

        GuardVariable actualBreakGuard;
        if (bodyInfo.cfTypes & CF_Break) {
            actualBreakGuard = breakGuard;
        } else if (bodyInfo.cfTypes & CF_Return) {
            // If there are no break statements we can use return guard as the break guard.
            actualBreakGuard = returnGuard;
        }

        if (actualBreakGuard) breakGuardBuilder(actualBreakGuard);

        // If the are no continue statements we can use break guard as the continue guard.
        if (actualBreakGuard && (bodyInfo.cfTypes & CF_Continue) == 0) {
            auto continueGuardVar = continueGuard.getVar();
            for (auto *user : llvm::make_early_inc_range(continueGuardVar->getUsers()))
                if (mlir::isa<P4HIR::AssignOp>(user)) rewriter.eraseOp(user);

            rewriter.replaceOp(continueGuardVar, actualBreakGuard.getVar());
        }

        CFInfo forInfo;
        if (bodyInfo.cfTypes & CF_Return) {
            // Only returns affect execution outside the loop.
            forInfo.cfTypes = CF_Return;
            forInfo.execType = ET_Multiple;
        }

        breakGuard = oldBreakGuard;
        continueGuard = oldContinueGuard;

        return forInfo;
    }

    static mlir::Block *getSingleBlock(mlir::Region &region) {
        assert(region.hasOneBlock() && "Cannot have multiple blocks in region");
        return &region.front();
    }

    // Transform an operation.
    CFInfo visitOp(mlir::Block *block, mlir::Operation *op) {
        if (auto softReturnOp = mlir::dyn_cast<P4HIR::SoftReturnOp>(op)) {
            rewriter.setInsertionPoint(op);
            returnGuard.assign(rewriter, op->getLoc(), false);

            if (isInLoop()) {
                breakGuard.assign(rewriter, op->getLoc(), false);
                continueGuard.assign(rewriter, op->getLoc(), false);
            }

            if (returnVar)
                rewriter.create<P4HIR::AssignOp>(softReturnOp.getLoc(), softReturnOp.getOperand(0),
                                                 returnVar);

            rewriter.eraseOp(op);
            return CFInfo(CF_Return);
        }

        if (mlir::isa<P4HIR::SoftBreakOp>(op)) {
            assert(isInLoop() && "Unexpected break outside of loop");
            rewriter.setInsertionPoint(op);
            breakGuard.assign(rewriter, op->getLoc(), false);
            continueGuard.assign(rewriter, op->getLoc(), false);
            rewriter.eraseOp(op);
            return CFInfo(CF_Break);
        }

        if (mlir::isa<P4HIR::SoftContinueOp>(op)) {
            assert(isInLoop() && "Unexpected continue outside of loop");
            rewriter.setInsertionPoint(op);
            continueGuard.assign(rewriter, op->getLoc(), false);
            rewriter.eraseOp(op);
            return CFInfo(CF_Continue);
        }

        if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(op))
            return visitBlock(getSingleBlock(scopeOp.getScopeRegion()));

        if (auto ifOp = mlir::dyn_cast<P4HIR::IfOp>(op)) {
            // Create else block if non existent.
            mlir::Region *elseRegion = &ifOp.getElseRegion();
            if (elseRegion->empty()) {
                mlir::Block *block = rewriter.createBlock(elseRegion, elseRegion->begin());
                rewriter.setInsertionPointToStart(block);
                rewriter.create<P4HIR::YieldOp>(ifOp.getLoc());
            }

            SmallVector<mlir::Block *, 2> blocks = {getSingleBlock(ifOp.getThenRegion()),
                                                    getSingleBlock(ifOp.getElseRegion())};
            return visitConditional(blocks);
        }

        if (auto switchOp = mlir::dyn_cast<P4HIR::SwitchOp>(op)) {
            // Create default case if non existent.
            if (!switchOp.getDefaultCase()) {
                rewriter.setInsertionPoint(switchOp.getBody().back().getTerminator());
                auto loc = switchOp.getLoc();
                rewriter.create<P4HIR::CaseOp>(
                    loc, mlir::ArrayAttr::get(rewriter.getContext(), {}),
                    P4HIR::CaseOpKind::Default,
                    [&](mlir::OpBuilder &b, mlir::Location) { b.create<P4HIR::YieldOp>(loc); });
            }

            auto blocks = llvm::map_to_vector(switchOp.cases(), [](P4HIR::CaseOp caseOp) {
                return getSingleBlock(caseOp.getCaseRegion());
            });
            return visitConditional(blocks);
        }

        if (auto forOp = mlir::dyn_cast<P4HIR::ForOp>(op)) {
            mlir::Block *bodyBlock = getSingleBlock(forOp.getBodyRegion());
            return visitLoop(forOp, bodyBlock, [&](GuardVariable &guard) {
                // Create a p4hir.ternary operation and only evaluate the original condition if
                // `guard` is enabled.
                mlir::Block *condBlock = getSingleBlock(forOp.getCondRegion());
                auto loc = forOp.getLoc();
                auto loopCond = mlir::cast<P4HIR::ConditionOp>(condBlock->getTerminator());

                rewriter.setInsertionPoint(loopCond);
                auto guardVal = rewriter.create<P4HIR::ReadOp>(loc, guard.getVar());
                auto newCond = rewriter.create<P4HIR::TernaryOp>(
                    loc, guardVal,
                    [&](mlir::OpBuilder &b, mlir::Location) {
                        b.create<P4HIR::YieldOp>(loc, mlir::ValueRange(loopCond.getCondition()));
                    },
                    [&](mlir::OpBuilder &b, mlir::Location) {
                        auto constFalse = b.create<P4HIR::ConstOp>(
                            loc, P4HIR::BoolAttr::get(rewriter.getContext(), false));
                        b.create<P4HIR::YieldOp>(loc, mlir::ValueRange(constFalse));
                    });

                auto opsToMove = llvm::make_range(condBlock->begin(), guardVal->getIterator());
                auto trueRegionTerminator =
                    getSingleBlock(newCond.getTrueRegion())->getTerminator();
                for (mlir::Operation &op : llvm::make_early_inc_range(opsToMove))
                    rewriter.moveOpBefore(&op, trueRegionTerminator);

                rewriter.modifyOpInPlace(loopCond, [&]() {
                    loopCond.getConditionMutable().assign(newCond.getResult());
                });

                // Also guard the updates region.
                mlir::Block *updatesBlock = getSingleBlock(forOp.getUpdatesRegion());
                guard.guardBlock(rewriter, loc, updatesBlock);
            });
        }

        if (auto forInOp = mlir::dyn_cast<P4HIR::ForInOp>(op)) {
            mlir::Block *bodyBlock = getSingleBlock(forInOp.getBodyRegion());
            return visitLoop(forInOp, bodyBlock, [&](GuardVariable &guard) {
                guard.guardBlock(rewriter, forInOp.getLoc(), bodyBlock);
            });
        }

        // Normal operation without nested control flow.
        [[maybe_unused]] auto checkNestedCF = [](mlir::Operation *op) -> mlir::WalkResult {
            if (mlir::isa<P4HIR::SoftReturnOp, P4HIR::SoftBreakOp, P4HIR::SoftContinueOp>(op))
                return WalkResult::interrupt();
            return WalkResult::advance();
        };
        assert(!op->walk(checkNestedCF).wasInterrupted() && "Unexpected control flow in operation");

        return CFInfo();
    }

    mlir::RewriterBase &rewriter;
    GuardVariable returnGuard;
    GuardVariable breakGuard;
    GuardVariable continueGuard;
    P4HIR::VariableOp returnVar;
};

void RemoveSoftCFPass::runOnOperation() {
    mlir::IRRewriter rewriter(&getContext());

    getOperation()->walk([&](mlir::Operation *op) {
        if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(op); funcOp && funcOp.getCallableRegion()) {
            RemoveSoftCF(rewriter).transform(
                op->getLoc(), funcOp.getCallableRegion(),
                mlir::cast<P4HIR::FuncType>(funcOp.getFunctionType()).getReturnType());
        } else if (auto controlApplyOp = mlir::dyn_cast<P4HIR::ControlApplyOp>(op);
                   controlApplyOp && !controlApplyOp.getBody().empty()) {
            RemoveSoftCF(rewriter).transform(op->getLoc(), &controlApplyOp.getBody(), {});
        }
    });
}

}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createRemoveSoftCFPass() {
    return std::make_unique<RemoveSoftCFPass>();
}
