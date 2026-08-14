// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <limits>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/ParserGraph.h"
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/ParserSymbolicExecution.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-parser-unroll"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_PARSERUNROLL
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

using BackEdge = std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>;

// Apply transition rewrites to a state's transitions and select cases.
static void applyTransitionRewrites(
    P4HIR::ParserOp parser, P4HIR::ParserStateOp state,
    const llvm::DenseMap<P4HIR::ParserStateOp, mlir::SymbolRefAttr> &rewrites) {
    if (rewrites.empty()) return;
    auto matches = [&](mlir::SymbolRefAttr ref) -> mlir::SymbolRefAttr {
        auto targetOp = parser.lookupSymbol<P4HIR::ParserStateOp>(ref);
        if (!targetOp) return {};
        auto it = rewrites.find(targetOp);
        if (it == rewrites.end()) return {};
        return it->second;
    };
    auto *terminator = state.getNextTransition();
    if (auto transition = mlir::dyn_cast<P4HIR::ParserTransitionOp>(terminator)) {
        if (auto replacement = matches(transition.getStateAttr()))
            transition.setStateAttr(replacement);
    } else if (auto selectOp = mlir::dyn_cast<P4HIR::ParserTransitionSelectOp>(terminator)) {
        for (auto selectCase : selectOp.selects()) {
            if (auto replacement = matches(selectCase.getStateAttr()))
                selectCase.setStateAttr(replacement);
        }
    }
}

struct DfsFrame {
    P4HIR::ParserStateOp state;
    llvm::SmallVector<P4HIR::ParserStateOp> nexts;
    unsigned nextIdx = 0;
};

// DFS from a particular state.
static void dfsFindBackEdges(
    P4HIR::ParserStateOp seed, llvm::DenseSet<P4HIR::ParserStateOp> &visited,
    llvm::function_ref<void(P4HIR::ParserStateOp, P4HIR::ParserStateOp)> onBackEdge) {
    if (visited.contains(seed)) return;
    llvm::DenseSet<P4HIR::ParserStateOp> onStack;
    llvm::SmallVector<DfsFrame> stack;

    visited.insert(seed);
    onStack.insert(seed);
    stack.push_back({seed, llvm::to_vector(seed.getNextStates()), 0});

    while (!stack.empty()) {
        bool descended = false;
        while (stack.back().nextIdx < stack.back().nexts.size()) {
            P4HIR::ParserStateOp next = stack.back().nexts[stack.back().nextIdx++];
            if (onStack.contains(next)) {
                onBackEdge(stack.back().state, next);
            } else if (!visited.contains(next)) {
                visited.insert(next);
                onStack.insert(next);
                stack.push_back({next, llvm::to_vector(next.getNextStates()), 0});
                descended = true;
                break;
            }
        }
        if (!descended) {
            onStack.erase(stack.back().state);
            stack.pop_back();
        }
    }
}

// Collect all back edges, warning about cycles unreachable from @start.
static llvm::SmallVector<BackEdge> findBackEdges(P4HIR::ParserOp parser) {
    llvm::SmallVector<BackEdge> result;

    auto startState = parser.getStartState();
    if (!startState) return result;

    llvm::DenseSet<P4HIR::ParserStateOp> visited;
    dfsFindBackEdges(startState, visited,
                     [&](P4HIR::ParserStateOp source, P4HIR::ParserStateOp dest) {
                         result.push_back({source, dest});
                     });

    for (auto stateOp : parser.states()) {
        if (visited.contains(stateOp) || stateOp.isTerminal()) continue;
        bool warned = false;
        dfsFindBackEdges(stateOp, visited,
                         [&](P4HIR::ParserStateOp /*source*/, P4HIR::ParserStateOp dest) {
                             if (warned) return;
                             warned = true;
                             dest.emitWarning()
                                 << "parser state '" << dest.getName()
                                 << "' is unreachable from @start but is the head "
                                    "of a cycle; parser-unroll will not process it";
                         });
    }

    return result;
}

// Map each cyclic state to its SCC members via llvm::scc_iterator.
static llvm::DenseMap<P4HIR::ParserStateOp, llvm::DenseSet<P4HIR::ParserStateOp>> collectSCCs(
    P4HIR::ParserOp parser) {
    llvm::DenseMap<P4HIR::ParserStateOp, llvm::DenseSet<P4HIR::ParserStateOp>> result;
    for (auto it = llvm::scc_begin(parser); !it.isAtEnd(); ++it) {
        const std::vector<mlir::Operation *> &component = *it;
        if (component.size() == 1 && !it.hasCycle()) continue;
        llvm::DenseSet<P4HIR::ParserStateOp> members;
        for (auto *op : component) {
            auto stateOp = mlir::cast<P4HIR::ParserStateOp>(op);
            if (!stateOp.isTerminal()) members.insert(stateOp);
        }
        for (auto stateOp : members) result[stateOp] = members;
    }
    return result;
}

struct LoopInfo {
    P4HIR::ParserStateOp head;
    llvm::DenseSet<P4HIR::ParserStateOp> members;
    llvm::SmallVector<StackAccess> combinedAccesses;
};

struct StateInfo {
    // Index into SCCInfo::loops, or -1 if not in any SCC.
    int loopIndex = -1;
    llvm::SmallVector<StackAccess> relevantStacks;
};

struct SCCInfo {
    llvm::SmallVector<LoopInfo> loops;
    llvm::DenseMap<P4HIR::ParserStateOp, StateInfo> stateInfo;
    llvm::DenseSet<P4HIR::ParserStateOp> rejectedMembers;

    bool empty() const { return loops.empty(); }
    const LoopInfo *loopOf(P4HIR::ParserStateOp state) const {
        auto it = stateInfo.find(state);
        if (it == stateInfo.end() || it->second.loopIndex < 0) return nullptr;
        return &loops[it->second.loopIndex];
    }
    llvm::ArrayRef<StackAccess> getRelevantStacks(P4HIR::ParserStateOp state) const {
        auto it = stateInfo.find(state);
        return (it != stateInfo.end()) ? llvm::ArrayRef(it->second.relevantStacks)
                                       : llvm::ArrayRef<StackAccess>{};
    }
};

// Loop candidate
struct PendingSCC {
    P4HIR::ParserStateOp head;
    llvm::DenseSet<P4HIR::ParserStateOp> members;
};

// Candidate SCCs per back-edge head, sorted.
static llvm::SmallVector<PendingSCC> collectPendingSCCs(
    P4HIR::ParserOp parser, llvm::ArrayRef<BackEdge> backEdges,
    const llvm::DenseMap<P4HIR::ParserStateOp, unsigned> &declarationPos) {
    auto sccMap = collectSCCs(parser);
    llvm::SmallVector<PendingSCC> pending;
    llvm::DenseSet<P4HIR::ParserStateOp> seenHead;
    for (auto &backEdge : backEdges) {
        auto head = backEdge.second;
        if (!seenHead.insert(head).second) continue;
        pending.push_back({head, sccMap.lookup(head)});
    }
    llvm::sort(pending, [&](const PendingSCC &lhs, const PendingSCC &rhs) {
        if (lhs.members.size() != rhs.members.size())
            return lhs.members.size() < rhs.members.size();
        return declarationPos.lookup(lhs.head) < declarationPos.lookup(rhs.head);
    });
    return pending;
}

// Combined stack accesses among states belonging to the SCC.
static llvm::SmallVector<StackAccess> combineSCCAccesses(
    P4HIR::ParserOp parser, P4HIR::ParserStateOp loopHead,
    const llvm::DenseSet<P4HIR::ParserStateOp> &sccSet, const AccessMap &stateAccesses) {
    llvm::SmallVector<StackAccess> combined;
    for (auto stateOp : parser.states()) {
        if (!sccSet.contains(stateOp)) continue;
        auto accessIt = stateAccesses.find(stateOp);
        assert(accessIt != stateAccesses.end() && "SCC state missing from stateAccesses");
        for (auto &access : accessIt->second) {
            auto *existing = llvm::find_if(
                combined, [&](const StackAccess &entry) { return entry.key == access.key; });
            if (existing == combined.end()) {
                combined.push_back(access);
                continue;
            }
            if (existing->size != access.size)
                loopHead.emitWarning()
                    << "header stack '" << renderStackId(access.key)
                    << "' appears with conflicting sizes in the same SCC; "
                       "unroll depth may be incorrect";
        }
    }
    return combined;
}

// Accept one SCC as an unrollable loop.
static void acceptLoopSCC(SCCInfo &scc, const PendingSCC &candidate, P4HIR::ParserOp parser,
                          const AccessMap &stateAccesses,
                          const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable,
                          unsigned maxUnrollDepth) {
    P4HIR::ParserStateOp loopHead = candidate.head;
    const auto &sccSet = candidate.members;

    auto reject = [&] { scc.rejectedMembers.insert(sccSet.begin(), sccSet.end()); };

    for (auto stateOp : sccSet) {
        if (untrackable.contains(stateOp)) {
            reject();
            return;
        }
    }

    auto combined = combineSCCAccesses(parser, loopHead, sccSet, stateAccesses);
    if (combined.empty()) {
        bool hasSelect = llvm::any_of(sccSet, [](P4HIR::ParserStateOp stateOp) {
            return mlir::isa<P4HIR::ParserTransitionSelectOp>(stateOp.getNextTransition());
        });
	// If the loop has no header stack and no select backedge, then it is infinite.
        if (!hasSelect) {
            loopHead.emitWarning()
                << "parser loop at state '" << loopHead.getName()
                << "' has no header stack operations and no select exit condition; "
                   "cannot unroll";
            reject();
            return;
        }
    } else {
        size_t minSize = std::numeric_limits<size_t>::max();
        for (auto &access : combined) minSize = std::min(minSize, access.size);
        if (minSize > maxUnrollDepth) {
            loopHead.emitWarning()
                << "parser loop at state '" << loopHead.getName() << "' would unroll to depth "
                << minSize << " (> " << maxUnrollDepth
                << "); skipping. Reduce header stack size or raise the limit.";
            reject();
            return;
        }
    }

    scc.loops.push_back({loopHead, sccSet, std::move(combined)});
}

// Collect variables used in transition_select args within loop states.
static llvm::DenseSet<StackId> collectSelectVars(P4HIR::ParserOp parser, const SCCInfo &scc,
                                                  StackNumbering &numbering) {
    llvm::DenseSet<StackId> selectVars;
    for (auto stateOp : parser.states()) {
        if (!scc.loopOf(stateOp)) continue;
        if (auto selectOp =
                mlir::dyn_cast<P4HIR::ParserTransitionSelectOp>(stateOp.getNextTransition()))
            for (mlir::Value arg : selectOp.getArgs())
                collectVarsInIndex(arg, selectVars, numbering);
    }
    return selectVars;
}

// Unique stack accesses reachable from start via DFS.
// accessesFor returns the accesses associated with each visited state.
static llvm::SmallVector<StackAccess> reachable(
    P4HIR::ParserStateOp start,
    llvm::function_ref<llvm::ArrayRef<StackAccess>(P4HIR::ParserStateOp)> accessesFor) {
    llvm::SmallVector<StackAccess> result;
    llvm::DenseSet<StackId> seenKeys;
    llvm::df_iterator_default_set<llvm::GraphTraits<P4HIR::ParserOp>::NodeRef> visited;
    for (auto stateOp : llvm::depth_first_ext(start, visited)) {
        auto current = mlir::cast<P4HIR::ParserStateOp>(stateOp);
        if (current.isTerminal()) continue;
        for (auto &access : accessesFor(current))
            if (seenKeys.insert(access.key).second) result.push_back(access);
    }
    return result;
}

// Per-state stacks that specialise clones.
static void computeStateInfo(P4HIR::ParserOp parser, SCCInfo &scc,
                             const AccessMap &stateAccesses) {
    for (auto [loopIndex, loop] : llvm::enumerate(scc.loops))
        for (auto stateOp : loop.members)
            scc.stateInfo[stateOp].loopIndex = static_cast<int>(loopIndex);

    bool acyclic = scc.loops.empty() && scc.rejectedMembers.empty();
    for (auto stateOp : parser.states()) {
        if (stateOp.isTerminal()) continue;
        if (scc.rejectedMembers.contains(stateOp)) continue;
        auto &info = scc.stateInfo[stateOp];
        if (acyclic) {
            info.relevantStacks =
                reachable(stateOp,
                          [&](P4HIR::ParserStateOp state) -> llvm::ArrayRef<StackAccess> {
                              auto it = stateAccesses.find(state);
                              if (it == stateAccesses.end()) return {};
                              return it->second;
                          });
        } else if (info.loopIndex >= 0) {
            info.relevantStacks = scc.loops[info.loopIndex].combinedAccesses;
        } else {
            info.relevantStacks =
                reachable(stateOp,
                          [&](P4HIR::ParserStateOp state) -> llvm::ArrayRef<StackAccess> {
                              auto *loop = scc.loopOf(state);
                              return loop ? llvm::ArrayRef(loop->combinedAccesses)
                                          : llvm::ArrayRef<StackAccess>{};
                          });
        }
    }
}

// Structurize loop data into SCC kernels.
static SCCInfo buildSCCInfo(
    P4HIR::ParserOp parser, llvm::ArrayRef<BackEdge> backEdges,
    const AccessMap &stateAccesses,
    const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable,
    const llvm::DenseMap<P4HIR::ParserStateOp, unsigned> &declarationPos,
    unsigned maxUnrollDepth) {
    SCCInfo scc;
    auto pending = collectPendingSCCs(parser, backEdges, declarationPos);
    for (auto &candidate : pending)
        acceptLoopSCC(scc, candidate, parser, stateAccesses, untrackable, maxUnrollDepth);
    computeStateInfo(parser, scc, stateAccesses);
    return scc;
}

struct PendingClone {
    unsigned originalDeclarationPos;
    P4HIR::ParserStateOp originalState;
    P4HIR::ParserStateOp bucketKey;
    SymbolicState *symbolicState;
};

// Creates list of states that needs to be cloned according to symbol execution.
static llvm::SmallVector<PendingClone> collectPendingClones(
    P4HIR::ParserOp parser, SymbolicResult &symbolicResult, const SCCInfo &scc,
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> &insertCursor) {
    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declarationPos;
    unsigned pos = 0;
    for (auto stateOp : parser.states()) {
        declarationPos[stateOp] = pos++;
        if (auto *loop = scc.loopOf(stateOp))
            insertCursor[loop->head] = stateOp;
    }

    llvm::SmallVector<PendingClone> clones;
    for (auto &symbolicState : symbolicResult.states) {
        symbolicState.cloneOp = symbolicState.state;
        if (symbolicState.callIndex == 0) continue;
        P4HIR::ParserStateOp originalState = symbolicState.state;
        auto declarationIt = declarationPos.find(originalState);
        assert(declarationIt != declarationPos.end() &&
               "cloned state missing from declarationPos - should be in parser.states()");
        P4HIR::ParserStateOp bucketKey;
        if (auto *loop = scc.loopOf(originalState)) {
            bucketKey = loop->head;
        } else {
            bucketKey = originalState;
            insertCursor.try_emplace(bucketKey, originalState);
        }
        clones.push_back({declarationIt->second, originalState, bucketKey, &symbolicState});
    }
    llvm::sort(clones, [](PendingClone &lhs, PendingClone &rhs) {
        if (lhs.symbolicState->callIndex != rhs.symbolicState->callIndex)
            return lhs.symbolicState->callIndex < rhs.symbolicState->callIndex;
        if (lhs.originalDeclarationPos != rhs.originalDeclarationPos)
            return lhs.originalDeclarationPos < rhs.originalDeclarationPos;
        return lhs.originalState.getOperation() < rhs.originalState.getOperation();
    });
    return clones;
}

// Make clones(identical for now)
static void createClones(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                          const SCCInfo &scc, StackNumbering &numbering) {
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> insertCursor;
    auto pendingClones = collectPendingClones(parser, symbolicResult, scc, insertCursor);

    mlir::OpBuilder builder(parser.getContext());
    for (auto &pending : pendingClones) {
        auto cursorIt = insertCursor.find(pending.bucketKey);
        assert(cursorIt != insertCursor.end() && "bucket key missing from insertCursor map");

        unsigned counter = pending.symbolicState->callIndex;
        auto uniqueName = mlir::SymbolTable::generateSymbolName<256>(
            pending.originalState.getSymName().str(),
            [&](llvm::StringRef candidate) {
                return parser.lookupSymbol(candidate) != nullptr;
            },
            counter);

        builder.setInsertionPointAfter(cursorIt->second.getOperation());
        mlir::IRMapping mapping;
        auto clone =
            mlir::cast<P4HIR::ParserStateOp>(builder.clone(*pending.originalState.getOperation(), mapping));
        clone.setSymName(uniqueName);
        for (auto [original, cloned] : mapping.getValueMap())
            if (auto it = numbering.valueIds.find(original); it != numbering.valueIds.end())
                numbering.valueIds[cloned] = it->second;
        pending.symbolicState->cloneOp = clone;
        cursorIt->second = clone;
    }
}

// Generate correct transitions between clones.
static void rewriteTransitions(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                               const SCCInfo &scc, mlir::SymbolRefAttr rejectRef,
                               StackNumbering &numbering) {
    using CloneKey = std::pair<P4HIR::ParserStateOp, unsigned>;
    llvm::DenseMap<CloneKey, P4HIR::ParserStateOp> cloneIndex;
    for (auto &symbolicState : symbolicResult.states)
        cloneIndex[{symbolicState.state, symbolicState.callIndex}] = symbolicState.cloneOp;

    struct StatePlan {
        P4HIR::ParserStateOp stateOp;
        llvm::DenseMap<P4HIR::ParserStateOp, mlir::SymbolRefAttr> rewrites;
    };
    llvm::SmallVector<StatePlan, 16> plans;
    plans.reserve(symbolicResult.states.size());

    for (auto &symbolicState : symbolicResult.states) {
        P4HIR::ParserStateOp stateOp = symbolicState.cloneOp;

        auto symbolicStateAccessIt = symbolicResult.accesses.find(symbolicState.state);
        assert(symbolicStateAccessIt != symbolicResult.accesses.end() &&
               "symbolicResult state missing from accesses map");
        IndexMap indexMapAfter = symbolicState.indexMap.advanced(symbolicStateAccessIt->second);
        ValueMap lookupValueMap =
            restrictValueMap(interpretState(
                                 symbolicState.state, symbolicState.entryValueMap,
                                 [](mlir::Operation *, const ValueMap &) {}, numbering),
                             symbolicResult.indexVars);

        StatePlan plan;
        plan.stateOp = stateOp;
        llvm::DenseSet<P4HIR::ParserStateOp> seenSuccessors;
        for (auto successor : stateOp.getNextStates()) {
            if (successor.isTerminal()) continue;
            if (scc.rejectedMembers.contains(successor)) continue;
            if (!seenSuccessors.insert(successor).second) continue;

            IndexMap lookupIndexMap = indexMapAfter.restrictTo(scc.getRelevantStacks(successor));

            auto successorIdx = symbolicResult.lookupSuccessor(
                successor, lookupIndexMap, lookupValueMap);
            assert(mlir::succeeded(successorIdx) &&
                   "BFS invariant violated: successor not found in visited map");
            if (*successorIdx && **successorIdx == 0) continue;

            if (*successorIdx) {
                auto cloneIt = cloneIndex.find({successor, **successorIdx});
                assert(cloneIt != cloneIndex.end() && "clone op missing from index");
                plan.rewrites[successor] = cloneIt->second.getSymbolRef();
            } else {
                plan.rewrites[successor] = rejectRef;
            }
        }
        if (!plan.rewrites.empty()) plans.push_back(std::move(plan));
    }

    for (auto &plan : plans) applyTransitionRewrites(parser, plan.stateOp, plan.rewrites);
}

// Substitute hs.nextIndex with constant indices.
static void substituteConstantIndices(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                      StackNumbering &numbering) {
    auto nextIndexKeyOf = [&](mlir::Value value) -> std::optional<StackId> {
        if (auto readOp = value.getDefiningOp<P4HIR::ReadOp>()) {
            if (auto nextIndexRef = readOp.getRef().getDefiningOp<P4HIR::StructFieldRefOp>();
                nextIndexRef && nextIndexRef.getFieldName() == "nextIndex")
                if (auto key = getStackId(nextIndexRef.getInput(), numbering); succeeded(key))
                    return *key;
        } else if (auto structExtract = value.getDefiningOp<P4HIR::StructExtractOp>()) {
            if (structExtract.getFieldName() == "nextIndex")
                if (auto key = getStackId(structExtract.getInput(), numbering); succeeded(key))
                    return *key;
        }
        return std::nullopt;
    };

    mlir::OpBuilder builder(parser.getContext());
    for (auto &symbolicState : symbolicResult.states) {
        if (!symbolicState.cloneOp) continue;

        llvm::DenseMap<StackId, unsigned> count;
        if (auto accessIt = symbolicResult.accesses.find(symbolicState.state);
            accessIt != symbolicResult.accesses.end())
            for (auto &access : accessIt->second) count[access.key] = access.count;

        llvm::DenseMap<StackId, unsigned> occurrence;
        std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &block) {
            for (auto &op : block) {
                if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op)) {
                    auto key = nextIndexKeyOf(elementRef.getIndex());
                    if (!key) continue;

                    int64_t value =
                        static_cast<int64_t>(symbolicState.indexMap.indexOf(*key)) +
                        occurrence[*key]++;
                    if (value < 0) continue;

                    auto idxType =
                        mlir::dyn_cast<P4HIR::BitsType>(elementRef.getIndex().getType());
                    if (!idxType) continue;

                    builder.setInsertionPoint(elementRef);
                    auto constOp = P4HIR::ConstOp::create(
                        builder, elementRef.getLoc(),
                        P4HIR::IntAttr::get(idxType, value));
                    elementRef.getIndexMutable().assign(constOp.getResult());
                } else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op)) {
                    for (auto &scopeBlock : scopeOp.getRegion())
                        visitBlock(scopeBlock);
                }
            }
        };
        visitBlock(*symbolicState.cloneOp.getBlock());
    }
}

// Fold non-constant stack indices that evaluate to a constant into ConstOps.
static void substituteExplicitIndices(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                      StackNumbering &numbering) {
    if (symbolicResult.indexVars.empty()) return;

    mlir::OpBuilder builder(parser.getContext());
    for (auto &symbolicState : symbolicResult.states) {
        if (!symbolicState.cloneOp) continue;

        interpretState(
            symbolicState.cloneOp, symbolicState.entryValueMap,
            [&](mlir::Operation *op, const ValueMap &valueMap) {
                mlir::Value idx;
                if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op))
                    idx = elementRef.getIndex();
                else if (auto arrayGet = mlir::dyn_cast<P4HIR::ArrayGetOp>(op))
                    idx = arrayGet.getIndex();
                else
                    return;

                if (matchPattern(idx, m_Constant())) return;

                auto idxType = mlir::dyn_cast<P4HIR::BitsType>(idx.getType());
                if (!idxType) return;

                auto value = foldToConstInt(idx, valueMap, numbering);
                if (failed(value) || value->isNegative()) return;

                builder.setInsertionPoint(op);
                auto constOp = P4HIR::ConstOp::create(
                    builder, op->getLoc(),
                    P4HIR::IntAttr::get(idxType, value->extOrTrunc(idxType.getWidth())));
                if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op))
                    elementRef.getIndexMutable().assign(constOp.getResult());
                else
                    mlir::cast<P4HIR::ArrayGetOp>(op).getIndexMutable().assign(constOp.getResult());
            },
            numbering);
    }
}

static mlir::SymbolRefAttr createOOBRejectState(P4HIR::ParserOp parser) {
    auto *context = parser.getContext();
    auto startState = parser.getStartState();
    mlir::IRRewriter rewriter(context);
    rewriter.setInsertionPoint(parser.getBody().front().getTerminator());
    auto state = IRUtils::createSubState(rewriter, startState, "outOfBound");
    rewriter.setInsertionPointToStart(state.getBlock());
    auto errorType = P4HIR::ErrorType::get(
        context, mlir::ArrayAttr::get(context, {mlir::StringAttr::get(context, "StackOutOfBounds")}));
    P4HIR::ParserRejectOp::create(rewriter, parser.getLoc(),
        P4HIR::ErrorCodeAttr::get(errorType, mlir::StringAttr::get(context, "StackOutOfBounds")));
    return state.getSymbolRef();
}

// Clone, substitute indices, and rewrite transitions into the unrolled parser.
static LogicalResult materializeUnrolled(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                         const SCCInfo &scc, StackNumbering &numbering) {
    if (symbolicResult.states.empty()) return success();

    bool hasOOB = llvm::any_of(symbolicResult.visitedMap,
                               [](const auto &entry) { return !entry.second; });

    mlir::SymbolRefAttr rejectRef;
    if (hasOOB)
        rejectRef = createOOBRejectState(parser);

    createClones(parser, symbolicResult, scc, numbering);
    substituteConstantIndices(parser, symbolicResult, numbering);
    substituteExplicitIndices(parser, symbolicResult, numbering);
    rewriteTransitions(parser, symbolicResult, scc, rejectRef, numbering);
    return success();
}

struct ParserUnroll : public impl::ParserUnrollBase<ParserUnroll> {
    using ParserUnrollBase::ParserUnrollBase;

    void runOnOperation() override {
        getOperation()->walk([&](P4HIR::ParserOp parser) {
            LLVM_DEBUG(llvm::dbgs() << "\n=== Parser Unroll: " << parser.getName() << " ===\n");

            auto backEdges = findBackEdges(parser);
            LLVM_DEBUG(llvm::dbgs() << "  back edges found: " << backEdges.size() << "\n");

            StackNumbering numbering;
            canonicalizeParserVariables(parser, numbering);
            AccessMap stateAccesses;
            llvm::DenseSet<P4HIR::ParserStateOp> untrackable;
            llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declarationPos;
            unsigned position = 0;
            for (auto stateOp : parser.states()) {
                declarationPos[stateOp] = position++;
                auto accesses = computeStackAccesses(stateOp, numbering);
                if (failed(accesses)) {
                    untrackable.insert(stateOp);
                    stateAccesses[stateOp] = {};
                } else {
                    stateAccesses[stateOp] = std::move(*accesses);
                }
            }

            auto scc = buildSCCInfo(parser, backEdges, stateAccesses, untrackable,
                                    declarationPos, maxUnrollDepth);
            LLVM_DEBUG({
                unsigned memberCount = 0;
                for (auto &loop : scc.loops) memberCount += loop.members.size();
                llvm::dbgs() << "  SCC members: " << memberCount << " across "
                             << scc.loops.size() << " loop(s)\n";
            });

            auto selectVars = collectSelectVars(parser, scc, numbering);

            auto symbolicResult =
                runSymbolicExecution(
                    parser, std::move(stateAccesses), numbering,
                    [&](P4HIR::ParserStateOp state) { return scc.getRelevantStacks(state); },
                    symbolicExecutionLimit, scc.rejectedMembers, selectVars);
            if (failed(materializeUnrolled(parser, symbolicResult, scc, numbering)))
                signalPassFailure();
        });
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createParserUnrollPass() { return std::make_unique<ParserUnroll>(); }

}  // namespace P4::P4MLIR
