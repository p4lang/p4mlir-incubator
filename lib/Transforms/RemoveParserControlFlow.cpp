#include "llvm/ADT/SmallString.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-parser-control-flow"

using namespace mlir;
using namespace llvm;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVEPARSERCONTROLFLOW
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR::P4HIR;

namespace {
struct RemoveParserControlFlowPass
    : public P4::P4MLIR::impl::RemoveParserControlFlowBase<RemoveParserControlFlowPass> {
    void runOnOperation() override;
};

struct DoRemoveParserControlFlowPattern : public OpRewritePattern<IfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(IfOp ifOp, PatternRewriter &rewriter) const override {
        auto stateOp = dyn_cast<ParserStateOp>(ifOp->getParentOp());
        if (!stateOp)
            return failure();

        auto parserOp = stateOp->getParentOfType<ParserOp>();
        if (!parserOp)
            return failure();

        bool madeChanges = false;
        if (hoistVariablesToParserRegion(ifOp.getThenRegion(), stateOp, rewriter)) madeChanges = true;
        if (!ifOp.getElseRegion().empty() 
            && hoistVariablesToParserRegion(ifOp.getElseRegion(), stateOp, rewriter)) madeChanges = true;

        // Create the new states true, false, join
        Location loc = ifOp.getLoc();
        SymbolTable symbolTable(stateOp->getParentOp());
        rewriter.setInsertionPointAfter(ifOp->getParentOp());
        // Create merge state
        auto mergeState = createStateOpFromVector(rewriter, symbolTable, collectOperationAfter(ifOp.getOperation()), Twine(stateOp.getName() + "_if_join"), loc);

        // Create true state
        rewriter.setInsertionPointAfter(ifOp->getParentOp());
        auto trueState = createStateOp(rewriter, symbolTable, Twine(stateOp.getName() + "_if_true"), loc);
        rewriter.cloneRegionBefore(ifOp.getThenRegion(), trueState->getRegion(0), trueState->getRegion(0).end());
        addTransitionOp(rewriter, trueState, mergeState.getName());

        // Create else state
        if (!ifOp.getElseRegion().empty()) {
            rewriter.setInsertionPointAfter(ifOp->getParentOp());
            auto falseState = createStateOp(rewriter, symbolTable, Twine(stateOp.getName() + "_if_false"), loc);
            rewriter.cloneRegionBefore(ifOp.getElseRegion(), falseState->getRegion(0), falseState->getRegion(0).end());
            addTransitionOp(rewriter, falseState, mergeState.getName());
        }

        // Create SelectOp
        rewriter.setInsertionPointAfter(ifOp.getOperation());
        auto selectOp = rewriter.create<ParserTransitionSelectOp>(loc, ifOp.getCondition());
        auto selectBlock = &selectOp.getBody().emplaceBlock();
        rewriter.setInsertionPointToStart(selectBlock);
        // Create True case
        rewriter.create<ParserSelectCaseOp>(loc, [&](OpBuilder &builder, Location loc) {
            auto boolAttr = P4::P4MLIR::P4HIR::BoolAttr::get(builder.getContext(), true);
            auto constTrue = builder.create<ConstOp>(loc, boolAttr);
            builder.create<YieldOp>(loc, constTrue.getResult());
        }, SymbolRefAttr::get(rewriter.getContext(), parserOp.getName(),
                              {FlatSymbolRefAttr::get(rewriter.getContext(), trueState.getName())}));

        // Create False case
        if (!ifOp.getElseRegion().empty()) {
            rewriter.create<ParserSelectCaseOp>(loc, [&](OpBuilder &builder, Location loc) {
                auto boolAttr = P4::P4MLIR::P4HIR::BoolAttr::get(builder.getContext(), true);
                auto constTrue = builder.create<ConstOp>(loc, boolAttr);
                builder.create<YieldOp>(loc, constTrue.getResult());
            }, SymbolRefAttr::get(rewriter.getContext(), parserOp.getName(),
                              {FlatSymbolRefAttr::get(rewriter.getContext(), trueState.getName())}));
        }
        rewriter.eraseOp(ifOp);
        return madeChanges ? success() : failure();
    }

private:
    bool hoistVariablesToParserRegion(Region &srcRegion, ParserStateOp &stateOp,
                                               PatternRewriter &rewriter) const {
        bool madeChanges = false;
        for (auto &block : srcRegion) {
            for (auto &op : llvm::make_early_inc_range(block)) {
                for (auto operand : op.getOperands()) {
                    // if the var is used inside the if op and is defined and
                    // at the top of the state region, then move it to the
                    // begining of the parser op.
                    auto defOp = operand.getDefiningOp();
                    if (!defOp) continue;
                    if (defOp->getParentRegion() == &stateOp->getRegion(0)) {
                        rewriter.setInsertionPointToStart(&stateOp->getParentOp()->getRegion(0).front());
                        auto newOp = rewriter.clone(*defOp);
                        rewriter.replaceAllUsesWith(defOp->getResults(), newOp->getResults());
                        rewriter.eraseOp(defOp);
                        madeChanges = true;
                    }
                }
            }
        }
        return madeChanges;
    }

    static StringAttr generateUniqueSymbolName(PatternRewriter &rewriter, SymbolTable &symbolTable,
                                               StringRef baseName) {
        unsigned uniqueCounter = 0;
        auto uniqueChecker = [&](StringRef name) {
            return symbolTable.lookup(rewriter.getStringAttr(name)) != nullptr;
        };

        SmallString<32> uniqueName = SymbolTable::generateSymbolName<32>(baseName, uniqueChecker, uniqueCounter);
        return rewriter.getStringAttr(uniqueName);
    }

    ParserStateOp createStateOp(PatternRewriter &rewriter, SymbolTable &symbolTable, Twine stateBaseName, Location loc) const {
        
        SmallString<32> nameBuffer;
        stateBaseName.toVector(nameBuffer);
        StringAttr newStateName = generateUniqueSymbolName(rewriter, symbolTable, nameBuffer);
        return rewriter.create<ParserStateOp>(loc, newStateName, rewriter.getDictionaryAttr({}));
    }

    ParserStateOp createStateOpFromVector(PatternRewriter &rewriter, SymbolTable &symbolTable,
                                          SmallVector<Operation*> srcOps, Twine stateBaseName,
                                          Location loc) const {
        auto newState = createStateOp(rewriter, symbolTable, stateBaseName, loc);
        rewriter.setInsertionPointToStart(&newState->getRegion(0).emplaceBlock());
        for (auto *op : srcOps) {
            auto newOp = rewriter.clone(*op);
            rewriter.replaceAllUsesWith(op->getResults(), newOp->getResults());
            rewriter.eraseOp(op);
        }
        return newState;
    }

    SmallVector<Operation *> collectOperationAfter(Operation *op) const {
        SmallVector<Operation*> result;
        auto startItr = op->getIterator();
        ++startItr;
        for (auto opItr = startItr; opItr != op->getBlock()->end(); ++opItr)
            result.push_back(&*opItr);
        return result;
    }

    void addTransitionOp(PatternRewriter &rewriter, ParserStateOp &stateOp, StringRef nextStateName) const {
        auto parserOp = stateOp->getParentOfType<ParserOp>();
        if (!parserOp) return;

        if (stateOp.getBody().empty()) rewriter.createBlock(&stateOp.getBody());

        Block &block = stateOp.getBody().front();
        SymbolRefAttr nextStateRef 
            = SymbolRefAttr::get(rewriter.getContext(), parserOp.getName(),
                                 {FlatSymbolRefAttr::get(rewriter.getContext(), nextStateName)});
    
        Operation *terminator = block.getTerminator();
        if (terminator) rewriter.setInsertionPoint(terminator);
        else rewriter.setInsertionPointToEnd(&block);
        
        rewriter.create<ParserTransitionOp>(stateOp.getLoc(), nextStateRef);
        if (terminator) rewriter.eraseOp(terminator);
    }

};


void RemoveParserControlFlowPass::runOnOperation() {
    auto *module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<DoRemoveParserControlFlowPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createRemoveParserControlFlowPass() {
  return std::make_unique<RemoveParserControlFlowPass>();
}
