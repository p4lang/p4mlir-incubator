#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-inline-parsers"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_INLINEPARSERS
#define GEN_PASS_DEF_INLINECONTROLS
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

struct InlineControlsPass : public P4::P4MLIR::impl::InlineControlsBase<InlineControlsPass> {
    InlineControlsPass() = default;
    void runOnOperation() override;
};

struct InlineParsersPass : public P4::P4MLIR::impl::InlineParsersBase<InlineParsersPass> {
    InlineParsersPass() = default;
    void runOnOperation() override;
};

// Based on InliningUtils::remapInlinedLocations
struct RemapLocationHelper {
    RemapLocationHelper() {
        attrReplacer.addReplacement(
            [this](mlir::LocationAttr loc) -> std::pair<mlir::LocationAttr, mlir::WalkResult> {
                return {remapLoc(loc), mlir::WalkResult::skip()};
            });
    }

    void setCallerLoc(mlir::Location callerLocation) { callerLoc = callerLocation; }

    void updateLocations(mlir::Operation *op) {
        attrReplacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                                  /*replaceLocs=*/true);
    }

 private:
    mlir::LocationAttr remapLoc(mlir::Location loc) {
        auto [it, inserted] = mappedLocations.try_emplace(loc);
        if (inserted) {
            mlir::LocationAttr newLoc = stackLocations(loc, callerLoc.value());
            it->getSecond() = newLoc;
        }
        return it->second;
    }

    static mlir::LocationAttr stackLocations(mlir::Location callee, mlir::Location caller) {
        mlir::Location lastCallee = callee;
        llvm::SmallVector<mlir::CallSiteLoc> calleeInliningStack;
        while (auto nextCallSite = dyn_cast<mlir::CallSiteLoc>(lastCallee)) {
            calleeInliningStack.push_back(nextCallSite);
            lastCallee = nextCallSite.getCaller();
        }

        mlir::CallSiteLoc firstCallSite = mlir::CallSiteLoc::get(lastCallee, caller);
        for (mlir::CallSiteLoc currentCallSite : reverse(calleeInliningStack))
            firstCallSite = mlir::CallSiteLoc::get(currentCallSite.getCallee(), firstCallSite);

        return firstCallSite;
    }

    std::optional<mlir::Location> callerLoc;
    llvm::DenseMap<mlir::Location, mlir::LocationAttr> mappedLocations;
    mlir::AttrTypeReplacer attrReplacer;
};

template <typename OpTy>
struct InstantiateOpInlineHelper {
    InstantiateOpInlineHelper(mlir::RewriterBase &rewriter, P4HIR::InstantiateOp instOp)
        : rewriter(rewriter), instOp(instOp) {}

    mlir::LogicalResult init() {
        callee = instOp.getCallee<OpTy>();
        if (!callee) return mlir::failure();

        if (!instOp.getArgOperands().empty())
            return rewriter.notifyMatchFailure(instOp.getLoc(),
                                               "Cannot inline object with constructor arguments.");

        if (!callee.getCallableRegion()->hasOneBlock())
            return rewriter.notifyMatchFailure(instOp.getLoc(),
                                               "Cannot inline callable with multiple blocks.");

        calleeBlock = &callee.getCallableRegion()->front();
        caller = instOp->getParentOfType<OpTy>();
        setPrefix(instOp.getSymName());

        // Collect all ApplyOps for this InstantiateOp.
        caller.walk([&](P4HIR::ApplyOp applyOp) {
            if (applyOp.getInstantiateOp() == instOp) calls.push_back(applyOp);
        });

        remapLocationHelper.setCallerLoc(caller.getLoc());

        return mlir::success();
    }

    void adjustClonedOp(mlir::Operation *op) {
        op->walk([&](mlir::Operation *nestedOp) { updateNamesAndRefs(nestedOp); });
        remapLocationHelper.updateLocations(op);
    }

    mlir::Operation *inlineCloneOp(mlir::Operation *op, mlir::IRMapping &mapper) {
        mlir::Operation *newOp = rewriter.clone(*op, mapper);
        adjustClonedOp(newOp);
        return newOp;
    }

    mlir::StringAttr updateName(mlir::StringAttr attr) {
        if (attr.getValue().empty()) return attr;
        auto newName = (prefix + "." + attr.getValue()).str();
        return mlir::StringAttr::get(attr.getContext(), newName);
    }

    // Append the instantiated control's name as a prefix to names and refs in `op`, which is an
    // operation cloned from the callee object.
    void updateNamesAndRefs(mlir::Operation *op) {
        llvm::SmallVector<llvm::StringRef> attrsToUpdate = {"name", "sym_name"};
        for (auto attrName : attrsToUpdate) {
            if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
                op->setAttr(attrName, updateName(attr));
        }

        // Rewrite SymbolRefAttrs that have as `callee` as root.
        auto mod = caller->template getParentOfType<mlir::ModuleOp>();
        for (auto namedAttr : op->getAttrs()) {
            auto symbolAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(namedAttr.getValue());
            if (!symbolAttr) continue;

            auto root = symbolAttr.getRootReference();
            if (mod.template lookupSymbol<OpTy>(root) != callee) continue;

            llvm::SmallVector<mlir::FlatSymbolRefAttr> newNestedRefs;
            bool first = true;
            for (auto flatRefAttr : symbolAttr.getNestedReferences()) {
                if (first)
                    newNestedRefs.push_back(
                        mlir::FlatSymbolRefAttr::get(updateName(flatRefAttr.getAttr())));
                else
                    newNestedRefs.push_back(flatRefAttr);
                first = false;
            }

            auto newSymbol = mlir::SymbolRefAttr::get(caller.getSymNameAttr(), newNestedRefs);
            op->setAttr(namedAttr.getName(), newSymbol);
        }
    }

    void setPrefix(llvm::StringRef newPrefix) { prefix = newPrefix; }

    mlir::RewriterBase &rewriter;
    P4HIR::InstantiateOp instOp;
    OpTy caller;
    OpTy callee;
    mlir::Block *calleeBlock;
    llvm::SmallVector<P4HIR::ApplyOp, 4> calls;
    RemapLocationHelper remapLocationHelper;
    std::string prefix;
};

struct ParserOpInlineHelper : public InstantiateOpInlineHelper<P4HIR::ParserOp> {
    using InstantiateOpInlineHelper<P4HIR::ParserOp>::InstantiateOpInlineHelper;

    mlir::LogicalResult doInlining() {
        auto initStatus = init();
        if (initStatus.failed()) return initStatus;

        mlir::IRMapping mapper;
        llvm::SmallVector<mlir::Operation *> initOps;
        llvm::SmallVector<P4HIR::ParserStateOp> states;
        P4HIR::ParserTransitionOp calleeStartTransition;

        auto splitStateRewriters = llvm::map_to_vector(calls, [&](auto applyOp) {
            return std::make_unique<IRUtils::SplitStateRewriter>(rewriter, applyOp);
        });

        mlir::LogicalResult status = mlir::success();
        std::optional<mlir::Location> lastApplyLoc;
        for (auto &ssr : splitStateRewriters) {
            auto applyOp = mlir::cast<P4HIR::ApplyOp>(ssr->getSplitOp());
            lastApplyLoc = applyOp.getLoc();
            status = ssr->init();
            if (status.failed()) break;
        }

        if (status.failed()) {
            // Erase all newly introduced states.
            for (auto &ssr : splitStateRewriters) ssr->cancel();
            return rewriter.notifyMatchFailure(*lastApplyLoc, "Cannot inline subparser apply.");
        }

        rewriter.setInsertionPoint(instOp);
        for (mlir::Operation &op : *calleeBlock) {
            if (mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::InstantiateOp>(op)) {
                inlineCloneOp(&op, mapper);
            } else if (auto parserState = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
                states.push_back(parserState);
            } else if (auto transitionOp = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
                assert(!calleeStartTransition && "Expected only one top-level transition");
                calleeStartTransition = transitionOp;
            } else {
                // Other operations are for local variable initialization and need to be copied to
                // each apply call.
                initOps.push_back(&op);
            }
        }

        size_t index = 0;
        for (auto [applyOp, ssr] : llvm::zip_equal(calls, splitStateRewriters)) {
            mlir::IRMapping callMapper = mapper;

            // Map formal arguments to the apply's arguments.
            for (auto [formalArg, actualArg] :
                 llvm::zip_equal(calleeBlock->getArguments(), applyOp.getArgOperands())) {
                assert((formalArg.getType() == actualArg.getType()) &&
                       "Unexpected call argument type mismatch");
                callMapper.map(formalArg, actualArg);
            }

            std::string prefix = instOp.getSymName().str();
            if (calls.size() > 1) prefix += std::string("#") + std::to_string(++index);
            setPrefix(prefix);

            // Clone states.
            rewriter.setInsertionPointAfter(ssr->getPreState());

            for (P4HIR::ParserStateOp state : states) {
                auto newStateOp =
                    mlir::cast<P4HIR::ParserStateOp>(inlineCloneOp(state, callMapper));

                if (newStateOp.isAccept()) {
                    // Make the subparser's accept transition to the "post" state.
                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                    auto acceptOp =
                        mlir::cast<P4HIR::ParserAcceptOp>(newStateOp.getNextTransition());
                    rewriter.setInsertionPoint(acceptOp);
                    P4HIR::ParserStateOp postState = ssr->getPostState();
                    rewriter.replaceOpWithNewOp<P4HIR::ParserTransitionOp>(
                        acceptOp, postState.getSymbolRef());
                }
            }

            // Clone local variable initialization code in the "pre" state.
            rewriter.setInsertionPointToEnd(ssr->getPreState().getBlock());
            for (mlir::Operation *op : initOps) inlineCloneOp(op, callMapper);

            // Clone transition to callee's start state as terminator.
            inlineCloneOp(calleeStartTransition, callMapper);
        }

        for (auto &ssr : splitStateRewriters) ssr->finalize();
        rewriter.eraseOp(instOp);

        return mlir::success();
    }
};

struct ControlOpInlineHelper : public InstantiateOpInlineHelper<P4HIR::ControlOp> {
    using InstantiateOpInlineHelper<P4HIR::ControlOp>::InstantiateOpInlineHelper;

    mlir::LogicalResult doInlining() {
        auto initStatus = init();
        if (initStatus.failed()) return initStatus;

        struct ControlLocalInfo {
            // The underlying storage for the cloned control local.
            P4HIR::VariableOp localStorage;
            // The original value that was assigned to the callee's control local.
            mlir::Value origVal;
            // The new value that should be used after inlining.
            mlir::Value newVal;
            // True iff `localStorage` is an existing variable cloned from callee.
            bool isStorageOrigVar;
        };

        mlir::IRMapping mapper;
        llvm::MapVector<mlir::StringAttr, ControlLocalInfo> controlLocalMap;
        llvm::SmallVector<mlir::Operation *> initOps;
        mlir::Region *calleeControlApplyRegion;

        rewriter.setInsertionPoint(instOp);
        for (mlir::Operation &op : *calleeBlock) {
            if (mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::FuncOp, P4HIR::TableOp,
                          P4HIR::InstantiateOp>(op)) {
                auto *newOp = inlineCloneOp(&op, mapper);

                if (mlir::isa<P4HIR::FuncOp>(newOp)) {
                    // Introduce new reads for control locals that were promoted.
                    newOp->walk([&](P4HIR::SymToValueOp symbolRefOp) {
                        auto controlLocalSym = symbolRefOp.getDecl().getLeafReference();
                        auto it = controlLocalMap.find(controlLocalSym);
                        assert((it != controlLocalMap.end()) &&
                               "Control local must be declared before use");
                        auto &info = it->second;
                        bool addedRef = !mlir::isa<P4HIR::ReferenceType>(info.origVal.getType());
                        if (addedRef) {
                            mlir::OpBuilder::InsertionGuard guard(rewriter);
                            rewriter.setInsertionPoint(symbolRefOp);
                            auto newSymbolRefOp = rewriter.create<P4HIR::SymToValueOp>(
                                symbolRefOp.getLoc(), info.localStorage.getType(),
                                symbolRefOp.getDecl());
                            auto readOp = rewriter.create<P4HIR::ReadOp>(
                                symbolRefOp.getLoc(), newSymbolRefOp.getResult());
                            rewriter.replaceOp(symbolRefOp, readOp);
                        }
                    });
                }
            } else if (auto controlLocalOp = mlir::dyn_cast<P4HIR::ControlLocalOp>(op)) {
                // Once a control is inlined the values of control local ops need to be updated at
                // each call site. Hence we need to create adjusted control locals with mutable
                // variables as underlying storage.
                ControlLocalInfo info;
                info.origVal = controlLocalOp.getVal();
                P4HIR::ControlLocalOp newControlLocal;

                if (controlLocalOp.getVal().getDefiningOp<P4HIR::VariableOp>()) {
                    // Don't create a backing variable if there was one already.
                    newControlLocal = mlir::cast<P4HIR::ControlLocalOp>(inlineCloneOp(&op, mapper));
                    info.isStorageOrigVar = true;
                    info.localStorage = newControlLocal.getVal().getDefiningOp<P4HIR::VariableOp>();
                } else {
                    auto newVarType = controlLocalOp.getVal().getType();
                    // If a control local was read-only we need to promote it to read-write.
                    if (!mlir::isa<P4HIR::ReferenceType>(newVarType))
                        newVarType = P4HIR::ReferenceType::get(newVarType);
                    auto newVarName =
                        (instOp.getSymName() + "." + controlLocalOp.getSymName() + "_var").str();
                    info.isStorageOrigVar = false;
                    info.localStorage = rewriter.create<P4HIR::VariableOp>(controlLocalOp.getLoc(),
                                                                           newVarType, newVarName);
                    newControlLocal = rewriter.create<P4HIR::ControlLocalOp>(
                        controlLocalOp.getLoc(), controlLocalOp.getSymName(), info.localStorage);
                    adjustClonedOp(newControlLocal);
                }

                [[maybe_unused]] auto [it, ins] =
                    controlLocalMap.insert({newControlLocal.getSymNameAttr(), info});
                assert(ins && "Expected all inlined names to be unique");
            } else if (auto controlApplyOp = mlir::dyn_cast<P4HIR::ControlApplyOp>(op)) {
                calleeControlApplyRegion = &controlApplyOp.getBody();
            } else {
                // Other operations are for local variable initialization and need to be copied to
                // each apply call.
                initOps.push_back(&op);
            }
        }

        for (auto applyOp : calls) {
            mlir::IRMapping callMapper = mapper;
            rewriter.setInsertionPoint(applyOp);

            // Map formal arguments to the apply's arguments.
            for (auto [formalArg, actualArg] :
                 llvm::zip_equal(calleeBlock->getArguments(), applyOp.getArgOperands())) {
                assert((formalArg.getType() == actualArg.getType()) &&
                       "Unexpected call argument type mismatch");
                callMapper.map(formalArg, actualArg);
            }

            // Clone local variable initialization code.
            for (mlir::Operation *op : initOps) inlineCloneOp(op, callMapper);

            // Initialize cloned p4hir.control_locals from scope-local variables and values.
            for (auto &[sym, info] : controlLocalMap) {
                // If we didn't introduce an additional variable, then this local should be
                // initialized by operations in `initOps`.
                if (info.isStorageOrigVar) continue;

                assert(callMapper.contains(info.origVal) &&
                       "Missing mapping for control local value");
                info.newVal = callMapper.lookup(info.origVal);

                if (mlir::isa<P4HIR::ReferenceType>(info.newVal.getType())) {
                    // If the argument is a reference type then we need to copy the new value to the
                    // underlying storage.
                    auto val = rewriter.create<P4HIR::ReadOp>(applyOp.getLoc(), info.newVal);
                    rewriter.create<P4HIR::AssignOp>(applyOp.getLoc(), val, info.localStorage);
                    // And use the underlying storage in the inlined control apply block.
                    callMapper.map(info.origVal, info.localStorage.getResult());
                } else {
                    // Otherwise it is a read-only argument. Assign it to the underlying storage so
                    // it can be accessed by actions. The updated value is already mapped for the
                    // control apply block.
                    rewriter.create<P4HIR::AssignOp>(applyOp.getLoc(), info.newVal,
                                                     info.localStorage);
                }
            }

            // Clone callee's control apply block.
            if (!calleeControlApplyRegion->empty())
                for (mlir::Operation &op : calleeControlApplyRegion->front())
                    inlineCloneOp(&op, callMapper);

            // Copy values from p4hir.control_locals back to scope-local variables.
            for (auto &[sym, info] : controlLocalMap) {
                // If we didn't introduce an additional variable, then this local should be updated
                // by callee's control action.
                if (info.isStorageOrigVar) continue;

                if (mlir::isa<P4HIR::ReferenceType>(info.newVal.getType())) {
                    auto updatedLocalVal =
                        rewriter.create<P4HIR::ReadOp>(applyOp.getLoc(), info.localStorage);
                    rewriter.create<P4HIR::AssignOp>(applyOp.getLoc(), updatedLocalVal,
                                                     info.newVal);
                }
            }

            rewriter.eraseOp(applyOp);
        }

        rewriter.eraseOp(instOp);

        return mlir::success();
    }
};

template<typename InlinerT>
static void inlineIteratively(mlir::ModuleOp mod) {
    mlir::IRRewriter rewriter(mod.getContext());

    // Inline iteratively until we cannot inline any more.
    bool madeChanges;
    do {
        madeChanges = false;

        llvm::SmallVector<P4HIR::InstantiateOp, 8> instOps;
        mod->walk(
            [&](P4HIR::InstantiateOp instOp) { instOps.push_back(instOp); });

        for (P4HIR::InstantiateOp instOp : instOps)
            if (InlinerT(rewriter, instOp).doInlining().succeeded())
                madeChanges = true;
    } while (madeChanges);
}

}  // namespace

void InlineParsersPass::runOnOperation() {
    inlineIteratively<ParserOpInlineHelper>(getOperation());
}

void InlineControlsPass::runOnOperation() {
    inlineIteratively<ControlOpInlineHelper>(getOperation());
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineParsersPass() {
    return std::make_unique<InlineParsersPass>();
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineControlsPass() {
    return std::make_unique<InlineControlsPass>();
}
