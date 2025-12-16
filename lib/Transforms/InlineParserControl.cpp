#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Dominance.h"
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
        while (auto nextCallSite = mlir::dyn_cast<mlir::CallSiteLoc>(lastCallee)) {
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
    using updateAttrNameCb =
        llvm::function_ref<std::string(mlir::Operation *, llvm::StringRef, llvm::StringRef)>;

    InstantiateOpInlineHelper(mlir::RewriterBase &rewriter, P4HIR::InstantiateOp instOp,
                              mlir::DominanceInfo *domInfo = nullptr)
        : rewriter(rewriter), instOp(instOp), domInfo(domInfo) {}

    mlir::LogicalResult init() {
        callee = instOp.getCallee<OpTy>();
        if (!callee) return mlir::failure();

        assert(callee.getCallableRegion()->hasOneBlock() &&
               "Expected single-block callable region");
        calleeBlock = &callee.getCallableRegion()->front();
        caller = instOp->getParentOfType<OpTy>();

        // Collect all ApplyOps for this InstantiateOp.
        caller.walk([&](P4HIR::ApplyOp applyOp) {
            if (applyOp.getInstantiateOp() == instOp) calls.push_back(applyOp);
        });

        remapLocationHelper.setCallerLoc(caller.getLoc());

        return mlir::success();
    }

    // If `op` is a constructor param constant of `callee` then create, map and return the
    // corresponding instantiated constant value from `instOp`, otherwise return null.
    mlir::Operation *inlineCloneCtorParamOp(mlir::Operation *op, updateAttrNameCb renameCb,
                                            mlir::IRMapping &mapper) {
        auto constOp = mlir::dyn_cast<P4HIR::ConstOp>(op);
        if (!constOp) return nullptr;

        auto ctorParam = mlir::dyn_cast<P4HIR::CtorParamAttr>(constOp.getValue());
        if (!ctorParam) return nullptr;

        if (ctorParam.getParent().getLeafReference() != callee.getSymName()) return nullptr;

        // Find the constant that corresponds to this ctor param from `instOp`.
        auto ctorInputs = callee.getCtorType().getInputs();
        auto it = llvm::find_if(ctorInputs, [&](const auto &input) {
            const auto &[paramName, paramType] = input;
            assert((paramName != ctorParam.getName() || paramType == ctorParam.getType()) &&
                   "Unexpected ctor param type");
            return paramName == ctorParam.getName();
        });
        assert((it != ctorInputs.end()) && "Cannot find ctor input");
        auto index = std::distance(ctorInputs.begin(), it);
        auto ctorArg = instOp.getArgOperands()[index];

        // Create and map the ctor param replacement.
        mlir::Operation *newOp = rewriter.clone(*ctorArg.getDefiningOp(), mapper);
        adjustClonedOp(newOp, renameCb);
        mapper.map(op->getResults(), newOp->getResults());
        return newOp;
    }

    // Clone and adjust an operation that is being inlined from `callee` to `caller`.
    mlir::Operation *inlineCloneOp(mlir::Operation *op, updateAttrNameCb renameCb,
                                   mlir::IRMapping &mapper) {
        if (auto *ctorParamConstOp = inlineCloneCtorParamOp(op, renameCb, mapper))
            return ctorParamConstOp;

        mlir::Operation *newOp = rewriter.clone(*op, mapper);
        adjustClonedOp(newOp, renameCb);
        return newOp;
    }

    // Adjust names, symbols and locations for an operation cloned from `callee` to `caller`.
    void adjustClonedOp(mlir::Operation *op, updateAttrNameCb renameCb) {
        op->walk([&](mlir::Operation *nestedOp) { updateNamesAndRefs(nestedOp, renameCb); });
        remapLocationHelper.updateLocations(op);
    }

    // Update the names and refs in `op` using `renameCb`.
    void updateNamesAndRefs(mlir::Operation *op, updateAttrNameCb renameCb) {
        auto updateName = [&](llvm::StringRef attrName, mlir::StringAttr attr) {
            if (attr.getValue().empty()) return attr;
            return rewriter.getStringAttr(renameCb(op, attrName, attr.getValue()));
        };

        llvm::SmallVector<llvm::StringRef> attrsToUpdate = {"name", "sym_name"};
        for (auto attrName : attrsToUpdate) {
            if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
                op->setAttr(attrName, updateName(attrName, attr));
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
                    newNestedRefs.push_back(mlir::FlatSymbolRefAttr::get(
                        updateName(namedAttr.getName(), flatRefAttr.getAttr())));
                else
                    newNestedRefs.push_back(flatRefAttr);
                first = false;
            }

            auto newSymbol = mlir::SymbolRefAttr::get(caller.getSymNameAttr(), newNestedRefs);
            op->setAttr(namedAttr.getName(), newSymbol);
        }
    }

    mlir::RewriterBase &rewriter;
    P4HIR::InstantiateOp instOp;
    mlir::DominanceInfo *domInfo;
    OpTy caller;
    OpTy callee;
    mlir::Block *calleeBlock;
    llvm::SmallVector<P4HIR::ApplyOp, 4> calls;
    RemapLocationHelper remapLocationHelper;
};

struct ParserOpInlineHelper : public InstantiateOpInlineHelper<P4HIR::ParserOp> {
    using InstantiateOpInlineHelper<P4HIR::ParserOp>::InstantiateOpInlineHelper;

    mlir::LogicalResult doInlining() {
        size_t applyIndex = 0;
        auto renameCb = [&](mlir::Operation *op, llvm::StringRef name,
                            llvm::StringRef value) -> std::string {
            llvm::SmallString<32> res = instOp.getSymName();
            if (calls.size() > 1)
                if (name == "state" || (name == "sym_name" && mlir::isa<P4HIR::ParserStateOp>(op)))
                    res += "#" + std::to_string(applyIndex);
            res += ".";
            res += value;
            return static_cast<std::string>(res);
        };

        if (init().failed()) return mlir::failure();

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
                inlineCloneOp(&op, renameCb, mapper);
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

        for (auto [applyOp, ssr] : llvm::zip_equal(calls, splitStateRewriters)) {
            mlir::IRMapping callMapper = mapper;

            // Map formal arguments to the apply's arguments.
            for (auto [formalArg, actualArg] :
                 llvm::zip_equal(calleeBlock->getArguments(), applyOp.getArgOperands())) {
                assert((formalArg.getType() == actualArg.getType()) &&
                       "Unexpected call argument type mismatch");
                callMapper.map(formalArg, actualArg);
            }

            // Clone states.
            rewriter.setInsertionPointAfter(ssr->getPreState());

            for (P4HIR::ParserStateOp state : states) {
                auto newStateOp =
                    mlir::cast<P4HIR::ParserStateOp>(inlineCloneOp(state, renameCb, callMapper));

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
            for (mlir::Operation *op : initOps) inlineCloneOp(op, renameCb, callMapper);

            // Clone transition to callee's start state as terminator.
            inlineCloneOp(calleeStartTransition, renameCb, callMapper);

            applyIndex++;
        }

        for (auto &ssr : splitStateRewriters) ssr->finalize();
        rewriter.eraseOp(instOp);

        return mlir::success();
    }
};

struct ControlOpInlineHelper : public InstantiateOpInlineHelper<P4HIR::ControlOp> {
    using InstantiateOpInlineHelper<P4HIR::ControlOp>::InstantiateOpInlineHelper;

    mlir::LogicalResult doInlining() {
        auto renameCb = [&](mlir::Operation *op, llvm::StringRef name,
                            llvm::StringRef value) -> std::string {
            return (instOp.getSymName() + "." + value).str();
        };

        if (init().failed()) return mlir::failure();

        unsigned argCount = callee.getNumArguments();
        bool hasApply = !calls.empty();
        llvm::SmallVector<mlir::StringAttr> passthroughBySymbol;
        llvm::SmallVector<mlir::Value> passthroughByValue;
        passthroughBySymbol.resize(argCount);
        passthroughByValue.resize(argCount);

        if (hasApply) {
            assert(domInfo && "Expected dominance info for control inlining");
            for (unsigned i = 0; i < argCount; i++) {
                mlir::Value rep = calls[0].getArgOperands()[i];

                bool allSame = llvm::all_equal(
                    llvm::map_range(calls, [&](auto apply) { return apply.getArgOperands()[i]; }));

                // If a value is equal across all applies then it is a candidate for
                // passthrough.
                if (allSame && domInfo->dominates(rep, instOp)) {
                    if (rep.getDefiningOp<P4HIR::ConstOp>()) {
                        // Promote constant to value passthrough.
                        passthroughByValue[i] = rep;
                    } else if (mlir::isa<mlir::BlockArgument>(rep)) {
                        // Promote argument to symbol passthrough.
                        passthroughByValue[i] = rep;

                        mlir::Block &callerBlock = caller.getBody().front();
                        // Try to find an existing control local in the caller for that value.
                        for (mlir::Operation &op : callerBlock) {
                            auto controlLocal = mlir::dyn_cast<P4HIR::ControlLocalOp>(op);
                            if (!controlLocal) continue;

                            if (controlLocal.getVal() == rep) {
                                passthroughBySymbol[i] = controlLocal.getSymNameAttr();
                                break;
                            }
                        }

                        // Otherwise create a new local.
                        if (!passthroughBySymbol[i]) {
                            rewriter.setInsertionPointToStart(&callerBlock);

                            auto getUniqueName = [&](mlir::StringAttr toRename) {
                                unsigned counter = 0;
                                return mlir::SymbolTable::generateSymbolName<256>(
                                    toRename,
                                    [&](llvm::StringRef candidate) {
                                        return caller.lookupSymbol(
                                                   rewriter.getStringAttr(candidate)) != nullptr;
                                    },
                                    counter);
                            };

                            std::string argNameStr;
                            if (auto argName = caller.getArgAttrOfType<mlir::StringAttr>(
                                    i, P4HIR::FuncOp::getParamNameAttrName()))
                                argNameStr = argName.str();
                            auto nameAttr = getUniqueName(rewriter.getStringAttr(
                                llvm::Twine("__local_") + caller.getSymName() + "_" + argNameStr));
                            auto local =
                                rewriter.create<P4HIR::ControlLocalOp>(rep.getLoc(), nameAttr, rep);
                            passthroughBySymbol[i] = local.getSymNameAttr();
                        }
                    }
                }

                // Specifically for locally instantiated externs we may need to look through
                // SymToValueOp.
                if (mlir::isa<P4HIR::ExternType>(rep.getType()) && !passthroughBySymbol[i]) {
                    auto getExternDecl = [](mlir::Value val) -> mlir::SymbolRefAttr {
                        if (auto symToVal = val.getDefiningOp<P4HIR::SymToValueOp>())
                            return symToVal.getDecl();
                        return {};
                    };

                    bool allSameDecl = llvm::all_equal(llvm::map_range(calls, [&](auto apply) {
                        return getExternDecl(apply.getArgOperands()[i]);
                    }));
                    if (allSameDecl)
                        if (auto externDecl = getExternDecl(rep))
                            passthroughBySymbol[i] = externDecl.getLeafReference();
                }

                // If an extern is used as an argument it must be the same in all applies and
                // can only be inlined with symbol passthrough.
                if (!passthroughBySymbol[i] && mlir::isa<P4HIR::ExternType>(rep.getType()))
                    return calls[0].emitOpError(
                        "Cannot use different extern argument in apply call of the same "
                        "control instance");
            }
        }

        struct ControlLocalInfo {
            // The underlying storage for the cloned control local.
            P4HIR::VariableOp localStorage;
            // The original value that was assigned to the callee's control local.
            mlir::Value origVal;
            // The new value that should be used after inlining.
            mlir::Value newVal;
            // A symbol referring to a control local that can be used instead.
            mlir::StringAttr passthroughSym;
            // True iff `localStorage` is a newly introduced variable that is not cloned.
            bool hasNewLocalStorage = false;
            // True if the control local was promoted from a value to a variable.
            bool promoted = false;
        };

        // Map that holds information for control locals of callee that are inlined.
        llvm::MapVector<mlir::StringAttr, ControlLocalInfo> controlLocalMap;
        // Top-level IR mapper.
        mlir::IRMapping mapper;
        // Holds callee's top-level initialization ops.
        llvm::SmallVector<mlir::Operation *> initOps;
        mlir::Region *calleeControlApplyRegion = nullptr;

        // Inline passthrough values directly.
        for (auto [formalArg, valPassthrough] :
             llvm::zip_equal(calleeBlock->getArguments(), passthroughByValue)) {
            if (valPassthrough) mapper.map(formalArg, valPassthrough);
        }

        mlir::Operation *callerControlApply = caller.getBody().back().getTerminator();
        rewriter.setInsertionPoint(instOp);
        for (mlir::Operation &op : *calleeBlock) {
            if (!hasApply) {
                if (mlir::isa<P4HIR::ConstOp, P4HIR::InstantiateOp>(op))
                    inlineCloneOp(&op, renameCb, mapper);
                continue;
            }

            if (mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::FuncOp, P4HIR::TableOp,
                          P4HIR::InstantiateOp>(op)) {
                mlir::Operation *newOp;

                if (mlir::isa<P4HIR::FuncOp, P4HIR::TableOp>(op)) {
                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPoint(callerControlApply);
                    newOp = inlineCloneOp(&op, renameCb, mapper);
                } else {
                    newOp = inlineCloneOp(&op, renameCb, mapper);
                }

                if (mlir::isa<P4HIR::FuncOp>(newOp)) {
                    // Adjust control locals after inlining.
                    newOp->walk([&](P4HIR::SymToValueOp symbolRefOp) {
                        mlir::OpBuilder::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPoint(symbolRefOp);

                        auto controlLocalSym = symbolRefOp.getDecl().getLeafReference();
                        auto it = controlLocalMap.find(controlLocalSym);
                        assert((it != controlLocalMap.end()) && "Cannot find control local info");
                        auto &info = it->second;

                        if (info.passthroughSym) {
                            // Replace symbol_ref with passthrough symbol.
                            auto *ctx = rewriter.getContext();
                            auto leafSymbol =
                                mlir::FlatSymbolRefAttr::get(ctx, info.passthroughSym);
                            auto newSymbolRef =
                                mlir::SymbolRefAttr::get(ctx, caller.getSymName(), {leafSymbol});
                            rewriter.replaceOpWithNewOp<P4HIR::SymToValueOp>(
                                symbolRefOp, symbolRefOp.getType(), newSymbolRef);
                        } else if (info.newVal) {
                            // Replace symbol_ref with passthrough value.
                            mlir::Value newVal = info.newVal;
                            if (auto constOp = info.newVal.getDefiningOp<P4HIR::ConstOp>())
                                newVal = rewriter.clone(*constOp)->getResult(0);
                            else
                                llvm_unreachable("Unknown type of value passthrough");

                            rewriter.replaceOp(symbolRefOp, newVal);
                        } else if (info.promoted) {
                            // Control local was promoted from value to reference.
                            // We need to introduce a read operation.
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
                auto newName = renameCb(controlLocalOp, "sym_name", controlLocalOp.getSymName());
                auto [it, ins] =
                    controlLocalMap.insert({rewriter.getStringAttr(newName), ControlLocalInfo{}});
                assert(ins && "Expected all inlined names to be unique");

                ControlLocalInfo &info = it->second;
                auto origVal = controlLocalOp.getVal();
                info.origVal = origVal;

                // Check for passthrough by symbol. We don't clone the control local in this case.
                if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(origVal)) {
                    if (auto passthroughSym = passthroughBySymbol[blockArg.getArgNumber()]) {
                        info.passthroughSym = passthroughSym;
                        continue;
                    }

                    if (auto passthroughVal = passthroughByValue[blockArg.getArgNumber()]) {
                        info.newVal = passthroughVal;
                        continue;
                    }
                }

                // Once a control is inlined the values of control local ops need to be updated at
                // each call site. Hence we need to ensure the inlined control locals use mutable
                // variables as the underlying storage.
                if (origVal.getDefiningOp<P4HIR::VariableOp>()) {
                    // Don't create a new backing variable if there was one already.
                    auto newControlLocal =
                        mlir::cast<P4HIR::ControlLocalOp>(inlineCloneOp(&op, renameCb, mapper));
                    info.localStorage = newControlLocal.getVal().getDefiningOp<P4HIR::VariableOp>();
                } else {
                    auto newVarType = origVal.getType();
                    // If a control local was read-only we need to promote it to read-write.
                    if (!mlir::isa<P4HIR::ReferenceType>(newVarType)) {
                        newVarType = P4HIR::ReferenceType::get(newVarType);
                        info.promoted = true;
                    }
                    auto newVarName =
                        (instOp.getSymName() + "." + controlLocalOp.getSymName() + "_var").str();
                    info.hasNewLocalStorage = true;
                    info.localStorage = rewriter.create<P4HIR::VariableOp>(controlLocalOp.getLoc(),
                                                                           newVarType, newVarName);
                    auto newControlLocal = rewriter.create<P4HIR::ControlLocalOp>(
                        controlLocalOp.getLoc(), controlLocalOp.getSymName(), info.localStorage);
                    adjustClonedOp(newControlLocal, renameCb);
                }
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
            for (mlir::Operation *op : initOps) inlineCloneOp(op, renameCb, callMapper);

            // Initialize cloned p4hir.control_locals from scope-local variables and values.
            for (auto &[sym, info] : controlLocalMap) {
                // If we didn't introduce an additional variable, then this local should be
                // initialized by operations in `initOps`.
                if (!info.hasNewLocalStorage) continue;

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
                    inlineCloneOp(&op, renameCb, callMapper);

            // Copy values from p4hir.control_locals back to scope-local variables.
            for (auto &[sym, info] : controlLocalMap) {
                // If we didn't introduce an additional variable, then this local should be updated
                // by callee's control action.
                if (!info.hasNewLocalStorage) continue;

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

template <typename InlinerT>
static void inlineIteratively(mlir::ModuleOp mod, mlir::DominanceInfo *domInfo = nullptr) {
    mlir::IRRewriter rewriter(mod.getContext());

    // Inline iteratively until we cannot inline any more.
    bool madeChanges;
    do {
        madeChanges = false;

        llvm::SmallVector<P4HIR::InstantiateOp, 8> instOps;
        mod->walk([&](P4HIR::InstantiateOp instOp) { instOps.push_back(instOp); });

        for (P4HIR::InstantiateOp instOp : instOps)
            if (InlinerT(rewriter, instOp, domInfo).doInlining().succeeded()) madeChanges = true;
    } while (madeChanges);
}

}  // namespace

void InlineParsersPass::runOnOperation() {
    inlineIteratively<ParserOpInlineHelper>(getOperation());
}

void InlineControlsPass::runOnOperation() {
    inlineIteratively<ControlOpInlineHelper>(getOperation(), &getAnalysis<mlir::DominanceInfo>());
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineParsersPass() {
    return std::make_unique<InlineParsersPass>();
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineControlsPass() {
    return std::make_unique<InlineControlsPass>();
}
