#include <iterator>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "synthesize-actions"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SYNTHESIZEACTIONS
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct SynthActionInControlApplyPattern : public OpRewritePattern<P4HIR::ControlOp> {
    using OpRewritePattern<P4HIR::ControlOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::ControlOp controlOp,
                                  PatternRewriter &rewriter) const override {
        auto isTopLevel = BMv2IR::isTopLevelControl(controlOp);
        if (failed(isTopLevel) || !isTopLevel.value()) return failure();
        if (controlOp.getBody().empty()) return failure();

        auto controlApply =
            cast<P4HIR::ControlApplyOp>(controlOp.getBody().front().getTerminator());

        // Convert IfOps inside control_apply to BMv2IR::IfOp, this makes it so the ops that compute
        // the conditional expression are now inside the BMv2IR::IfOp condition region, so we don't
        // encounter them when synthesizing actions.
        auto ifRes = controlApply.walk([&](P4HIR::IfOp ifOp) {
            // if ops for hit or miss are handled when lowering to BMv2 pipelines
            if (BMv2IR::isHitOrMissIf(ifOp)) return WalkResult::skip();
            if (failed(convertIfOp(ifOp, controlApply, rewriter))) return WalkResult::interrupt();
            return WalkResult::advance();
        });
        if (ifRes.wasInterrupted()) return failure();

        // Synthesize actions in nested regions, we skip the "condition" region in BMv2IR::IfOp ops
        // since we don't need to synth actions there (the corresponding JSON node will contain the
        // full expression)
        auto walkRes = controlApply.walk([&](Operation *op) {
            if (auto ifOp = dyn_cast<BMv2IR::IfOp>(op)) {
                auto thenOk = synthesizeActionsInRegion(ifOp.getThenRegion(), rewriter, controlOp);
                auto elseOk = synthesizeActionsInRegion(ifOp.getElseRegion(), rewriter, controlOp);
                if (failed(thenOk) || failed(elseOk)) return WalkResult::interrupt();
                return WalkResult::advance();
            }

            if (auto switchOp = dyn_cast<P4HIR::SwitchOp>(op)) return WalkResult::advance();

            for (auto &region : op->getRegions()) {
                if (failed(synthesizeActionsInRegion(region, rewriter, controlOp)))
                    return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (walkRes.wasInterrupted()) return failure();

        return success();
    }

 private:
    static LogicalResult convertIfOp(P4HIR::IfOp op, P4HIR::ControlApplyOp controlApply,
                                     PatternRewriter &rewriter) {
        auto name = BMv2IR::getUniqueNameInParentModule(op, "cond_node");

        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);

        auto newIf = rewriter.create<BMv2IR::IfOp>(op.getLoc(), name);
        newIf.getThenRegion().takeBody(op.getThenRegion());
        newIf.getElseRegion().takeBody(op.getElseRegion());
        auto replaceYield = [&](Region &region) {
            if (region.empty()) return;
            auto op = region.front().getTerminator();
            auto p4yield = cast<P4HIR::YieldOp>(op);
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(p4yield);
            rewriter.replaceOpWithNewOp<BMv2IR::YieldOp>(p4yield, p4yield.getArgs());
        };
        replaceYield(newIf.getThenRegion());
        replaceYield(newIf.getElseRegion());

        // Move ops the are used to compute the IfOp condition into the BMv2IR::IfOp condition
        // region. We assume that the leaf nodes in the expression
        // are Header Instances.
        // TODO: could there be other kinds of ops as leaf? Constants?
        SmallVector<Operation *> expressionOps;
        if (failed(getIfOpConditionOps(op.getLoc(), op.getCondition(), expressionOps)))
            return op.emitError("Error retrieving expression ops");
        auto &block = newIf.getConditionRegion().emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        for (auto op : llvm::reverse(expressionOps)) {
            rewriter.moveOpBefore(op, &block, block.end());
        }
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<BMv2IR::YieldOp>(op.getLoc(), expressionOps[0]->getResult(0));
        rewriter.replaceOp(op, newIf);

        return success();
    }

    static LogicalResult getIfOpConditionOps(Location loc, Value v, SmallVector<Operation *> &ops) {
        auto defOp = v.getDefiningOp();
        if (!defOp) return emitError(loc, "Expected defining operation");
        ops.push_back(defOp);
        if (isa<BMv2IR::SymToValueOp>(defOp)) return success();

        for (auto &operand : defOp->getOpOperands()) {
            if (failed(getIfOpConditionOps(loc, operand.get(), ops))) return failure();
        }
        return success();
    }

    static P4HIR::FuncOp getNewAction(PatternRewriter &rewriter, Location loc,
                                      P4HIR::ControlOp parent) {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&parent.getBody().front());
        unsigned counter = 0;
        auto name = rewriter.getStringAttr("dummy_action");
        auto uniqueName = SymbolTable::generateSymbolName<256>(
            name,
            [&](StringRef candidate) {
                return SymbolTable::lookupSymbolIn(parent, candidate) != nullptr;
            },
            counter);
        auto fTy =
            P4HIR::FuncType::get(ArrayRef<Type>{}, P4HIR::VoidType::get(rewriter.getContext()));
        auto newAction =
            rewriter.create<P4HIR::FuncOp>(loc, rewriter.getStringAttr(uniqueName), fTy);
        newAction.setAction(true);
        return newAction;
    }

    static FailureOr<P4HIR::FuncOp> synthesizeAction(ArrayRef<Operation *> ops,
                                                     PatternRewriter &rewriter,
                                                     P4HIR::ControlOp parent) {
        if (ops.empty()) return success();
        auto lastOp = ops.back();
        auto loc = lastOp->getLoc();
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(ops.front());
        if (lastOp->getNumResults() != 0)
            return emitError(loc, "Last op returning a value not supported");
        auto newAction = getNewAction(rewriter, loc, parent);
        auto &block = newAction.getBody().emplaceBlock();
        for (auto &op : ops) {
            // TODO: here we should check the op uses, and move only if there
            // are no uses or if all ops uses are contained in `ops`, otherwise we should
            // clone
            rewriter.moveOpBefore(op, &block, block.end());
        }
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<P4HIR::ReturnOp>(loc);
        return newAction;
    }

    static LogicalResult cloneOpOperandsFromOutsideRegion(Operation *op,
                                                          SmallVector<Operation *> &res,
                                                          PatternRewriter &rewriter) {
        auto parentOp = op->getParentOp();
        PatternRewriter::InsertionGuard guard(rewriter);

        for (auto &operand : op->getOpOperands()) {
            auto val = operand.get();
            auto defOp = val.getDefiningOp();
            if (!defOp) return op->emitError("Expected operands to come from other ops");
            if (defOp->getParentOp() == parentOp) continue;

            if (!isa<P4HIR::ConstOp>(defOp))
                return op->emitError("Unsupported op from outside region");
            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.clone(*defOp);
            rewriter.replaceOpUsesWithinBlock(defOp, newOp->getResults(), op->getBlock());
            res.push_back(newOp);
        }
        return success();
    }

    static LogicalResult synthesizeActionsInRegion(Region &region, PatternRewriter &rewriter,
                                                   P4HIR::ControlOp parent) {
        for (Block &block : region) {
            if (failed(synthesizeActionsInBlock(block, rewriter, parent))) return failure();
        }
        return success();
    }

    static LogicalResult synthesizeActionsInBlock(Block &block, PatternRewriter &rewriter,
                                                  P4HIR::ControlOp parent) {
        PatternRewriter::InsertionGuard guard(rewriter);
        SmallVector<Operation *> rawOps;

        auto doSynthAction = [&](Operation *opInsertPoint, Block *blockInsertPoint) {
            if (rawOps.empty()) return success();
            assert((opInsertPoint == nullptr) != (blockInsertPoint == nullptr) &&
                   "Only one insert point allowed");
            PatternRewriter::InsertionGuard guard(rewriter);
            auto maybeAction = synthesizeAction(rawOps, rewriter, parent);
            if (failed(maybeAction)) return failure();
            if (opInsertPoint) {
                rewriter.setInsertionPoint(opInsertPoint);
            } else {
                rewriter.setInsertionPointToEnd(blockInsertPoint);
            }
            auto action = maybeAction.value();
            auto retTy = P4HIR::VoidType::get(rewriter.getContext());
            auto ref = SymbolRefAttr::get(parent.getSymNameAttr(),
                                          {SymbolRefAttr::get(action.getSymNameAttr())});
            rewriter.create<P4HIR::CallOp>(rawOps.front()->getLoc(), ref, retTy, TypeRange{},
                                           ValueRange{});
            rawOps.clear();
            return success();
        };

        for (auto it = block.begin(); it != block.end();) {
            Operation &op = *it;
            if (isa<BMv2IR::IfOp, P4HIR::TableApplyOp, BMv2IR::YieldOp, P4HIR::YieldOp>(op)) {
                if (failed(doSynthAction(&op, nullptr))) return failure();
                if (isa<P4HIR::TableApplyOp>(op)) {
                    // If a table_apply is followed by a switch statement (or an if on hit/miss), we
                    // have to skip the extract -> switch ops because we want them in the
                    // control_apply for pipelines lowering
                    auto peekNext = std::next(it);
                    if (peekNext == block.end()) {
                        it++;
                        continue;
                    }
                    auto peekNextNext = std::next(peekNext);
                    if (peekNextNext == block.end()) {
                        it++;
                        continue;
                    }
                    if (isa<P4HIR::StructExtractOp>(*peekNext) &&
                        (isa<P4HIR::SwitchOp>(*peekNextNext) ||
                         BMv2IR::isHitOrMissIf(&*peekNextNext))) {
                        std::advance(it, 3);
                        continue;
                    } else {
                        it++;
                        continue;
                    }
                } else {
                    it++;
                }
            } else {
                if (failed(cloneOpOperandsFromOutsideRegion(&op, rawOps, rewriter)))
                    return failure();
                rawOps.push_back(&op);
                it++;
            }
        }
        if (!rawOps.empty()) {
            return doSynthAction(nullptr, &block);
        }
        return success();
    }
};

struct SynthesizeActionsPass
    : public P4::P4MLIR::impl::SynthesizeActionsBase<SynthesizeActionsPass> {
    void runOnOperation() override {
        auto &context = getContext();
        auto moduleOp = getOperation();
        RewritePatternSet patterns(&context);
        patterns.add<SynthActionInControlApplyPattern>(&context);
        walkAndApplyPatterns(moduleOp, std::move(patterns));
    }
};

}  // anonymous namespace
