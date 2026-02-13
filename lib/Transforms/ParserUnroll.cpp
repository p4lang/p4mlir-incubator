#include "p4mlir/Dialect/P4HIR/ParserGraph.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Transforms/Passes.h"
#include <deque>
#include <limits>

#define DEBUG_TYPE "p4hir-parser-unroll"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_PARSERUNROLL
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

static constexpr unsigned kDefaultMaxUnrollDepth = 64;
static constexpr unsigned kMaxBFSStateInstances = 100000;

// Core types — Definitions 2–5 of the algorithm.
// Definition 1 (the parser state graph) is P4HIR::ParserOp itself.

// Definition 5 / {HSp}: a header stack variable used in a state, plus its size.
struct StackAccess {
    std::string key;  // the stack variable (∈ {HSp})
    size_t size;      // OOB when M[key] >= size
};

// Definition 2: symbolic index map M = {stack_key → nextIndex}.
class IndexMap {
public:
    bool isOOBFor(const StackAccess &acc) const {
        auto it = data_.find(acc.key);
        return it != data_.end() && it->second >= acc.size;
    }
    bool isOOBForAny(llvm::ArrayRef<StackAccess> accesses) const {
        for (auto &acc : accesses)
            if (isOOBFor(acc)) return true;
        return false;
    }
    IndexMap advanced(llvm::ArrayRef<StackAccess> accesses) const {
        IndexMap result = *this;
        for (auto &acc : accesses)
            ++result.data_[acc.key];
        return result;
    }
    IndexMap restrictTo(llvm::ArrayRef<StackAccess> relevant) const {
        IndexMap result;
        for (auto &acc : relevant) {
            auto it = data_.find(acc.key);
            if (it != data_.end())
                result.data_[acc.key] = it->second;
        }
        return result;
    }
    bool operator<(const IndexMap &o) const { return data_ < o.data_; }

private:
    std::map<std::string, unsigned> data_;
};

// Definition 3 / Definition 4: dedup key encoding the pair (state, M).
struct VisitedKey {
    mlir::StringAttr name;
    IndexMap M;

    bool operator<(const VisitedKey &o) const {
        const void *a = name.getAsOpaquePointer();
        const void *b = o.name.getAsOpaquePointer();
        if (a != b) return a < b;
        return M < o.M;
    }
};

static inline bool isTerminal(P4HIR::ParserStateOp s) {
    return s.isAccept() || s.isReject();
}

static std::optional<size_t> stackSizeOf(mlir::Type type) {
    if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type))
        type = ref.getObjectType();
    if (auto hs = mlir::dyn_cast<P4HIR::HeaderStackType>(type))
        return hs.getArraySize();
    return std::nullopt;
}

static std::optional<std::string> getStackKey(mlir::Value v) {
    llvm::SmallVector<std::string, 4> reverseSegments;
    std::string base;

    while (true) {
        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
            base = ("$arg:" + llvm::Twine(arg.getArgNumber()) + "@" +
                    llvm::Twine::utohexstr(
                        reinterpret_cast<uintptr_t>(arg.getOwner())))
                       .str();
            break;
        }
        auto *defOp = v.getDefiningOp();
        if (!defOp) return std::nullopt;

        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(defOp)) {
            if (var.getName())
                base = ("$var:" + llvm::Twine(*var.getName())).str();
            else
                base = ("$anon:" +
                        llvm::Twine::utohexstr(
                            reinterpret_cast<uintptr_t>(var.getOperation())))
                           .str();
            break;
        }
        if (auto fr = mlir::dyn_cast<P4HIR::StructFieldRefOp>(defOp)) {
            reverseSegments.push_back(fr.getFieldName().str());
            v = fr.getInput();
            continue;
        }
        if (auto se = mlir::dyn_cast<P4HIR::StructExtractOp>(defOp)) {
            reverseSegments.push_back(se.getFieldName().str());
            v = se.getInput();
            continue;
        }
        if (auto rd = mlir::dyn_cast<P4HIR::ReadOp>(defOp)) {
            v = rd.getRef();
            continue;
        }
        return std::nullopt;
    }

    std::string result = std::move(base);
    for (auto it = reverseSegments.rbegin(); it != reverseSegments.rend(); ++it) {
        result += '.';
        result += *it;
    }
    return result;
}

static bool flowsToArrayElementRefIndex(mlir::Value root) {
    llvm::SmallPtrSet<mlir::Value, 8> seen;
    llvm::SmallVector<mlir::Value, 8> wl;
    wl.push_back(root);
    while (!wl.empty()) {
        mlir::Value cur = wl.pop_back_val();
        if (!seen.insert(cur).second) continue;
        for (mlir::OpOperand &use : cur.getUses()) {
            mlir::Operation *user = use.getOwner();
            if (auto aer = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(user)) {
                if (aer.getIndex() == use.get()) return true;
                continue;
            }
            if (mlir::isa<P4HIR::ReadOp, P4HIR::CastOp>(user)) {
                if (user->getNumResults() == 1)
                    wl.push_back(user->getResult(0));
            }
        }
    }
    return false;
}

// Definition 5 / ParserStructure: builds {HSp} for a single state.
static std::optional<llvm::SmallVector<StackAccess>>
computeStackAccesses(P4HIR::ParserStateOp state) {
    llvm::SmallVector<StackAccess> result;
    llvm::StringSet<> seen;
    bool unidentified = false;

    auto record = [&](mlir::Value input) {
        auto sz = stackSizeOf(input.getType());
        if (!sz || *sz == 0) return;
        auto key = getStackKey(input);
        if (!key) {
            unidentified = true;
            return;
        }
        if (!seen.insert(*key).second) return;
        result.push_back({std::move(*key), *sz});
    };

    state.walk([&](mlir::Operation *op) {
        if (auto fr = mlir::dyn_cast<P4HIR::StructFieldRefOp>(op)) {
            if (fr.getFieldName() == "nextIndex" &&
                flowsToArrayElementRefIndex(fr.getResult()))
                record(fr.getInput());
        } else if (auto se = mlir::dyn_cast<P4HIR::StructExtractOp>(op)) {
            if (se.getFieldName() == "nextIndex" &&
                flowsToArrayElementRefIndex(se.getResult()))
                record(se.getInput());
        } else if (auto elemRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op)) {
            mlir::Value idx = elemRef.getIndex();
            while (auto cast = idx.getDefiningOp<P4HIR::CastOp>())
                idx = cast.getSrc();
            if (mlir::matchPattern(idx, mlir::m_Constant())) return;

            auto *arrDef = elemRef.getInput().getDefiningOp();
            if (!arrDef) return;
            auto dataRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(arrDef);
            if (!dataRef || dataRef.getFieldName() != "data") return;

            record(dataRef.getInput());
        }
    });

    if (unidentified) {
        mlir::emitWarning(state.getLoc(),
            "cannot determine identity of header stack accessed in state '" +
            state.getName().str() + "'; not unrolling this loop");
        return std::nullopt;
    }
    return result;
}

static mlir::SymbolRefAttr makeStateRef(mlir::MLIRContext *ctx,
                                        llvm::StringRef stateName) {
    return mlir::FlatSymbolRefAttr::get(ctx, stateName);
}

static void
applyTransitionRewrites(P4HIR::ParserStateOp state,
                        const llvm::StringMap<mlir::SymbolRefAttr> &rewrites) {
    if (rewrites.empty()) return;
    auto matches = [&](mlir::SymbolRefAttr ref)
        -> mlir::SymbolRefAttr {
        auto it = rewrites.find(ref.getLeafReference().getValue());
        if (it == rewrites.end()) return {};
        return it->second;
    };
    state.walk([&](mlir::Operation *op) {
        if (auto t = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            if (auto rep = matches(t.getStateAttr())) t.setStateAttr(rep);
        } else if (auto c = mlir::dyn_cast<P4HIR::ParserSelectCaseOp>(op)) {
            if (auto rep = matches(c.getStateAttr())) c.setStateAttr(rep);
        }
    });
}

struct DfsFrame {
    P4HIR::ParserStateOp state;
    llvm::SmallVector<P4HIR::ParserStateOp> nexts;
    unsigned nextIdx = 0;
};

template <typename OnBackEdge>
static void dfsFindBackEdges(P4HIR::ParserStateOp seed,
                             llvm::DenseSet<P4HIR::ParserStateOp> &visited,
                             OnBackEdge onBackEdge) {
    if (visited.contains(seed)) return;
    llvm::DenseSet<P4HIR::ParserStateOp> onStack;
    llvm::SmallVector<DfsFrame> stack;

    visited.insert(seed);
    onStack.insert(seed);
    stack.push_back({seed, llvm::SmallVector<P4HIR::ParserStateOp>(
                               seed.getNextStates()), 0});

    while (!stack.empty()) {
        bool descended = false;
        while (stack.back().nextIdx < stack.back().nexts.size()) {
            P4HIR::ParserStateOp next =
                stack.back().nexts[stack.back().nextIdx++];
            if (onStack.contains(next)) {
                onBackEdge(stack.back().state, next);
            } else if (!visited.contains(next)) {
                visited.insert(next);
                onStack.insert(next);
                stack.push_back({next, llvm::SmallVector<P4HIR::ParserStateOp>(
                                            next.getNextStates()), 0});
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

static llvm::SmallVector<std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>>
findBackEdges(P4HIR::ParserOp parser) {
    llvm::SmallVector<std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>> result;

    auto startState = parser.getStartState();
    if (!startState) return result;

    llvm::DenseSet<P4HIR::ParserStateOp> visited;
    dfsFindBackEdges(startState, visited,
                     [&](P4HIR::ParserStateOp src, P4HIR::ParserStateOp dst) {
                         result.push_back({src, dst});
                     });

    for (auto s : parser.states()) {
        if (visited.contains(s) || isTerminal(s)) continue;
        bool warned = false;
        dfsFindBackEdges(s, visited,
                         [&](P4HIR::ParserStateOp /*src*/,
                             P4HIR::ParserStateOp dst) {
                             if (warned) return;
                             warned = true;
                             mlir::emitWarning(dst.getLoc(),
                                 "parser state '" + dst.getName().str() +
                                 "' is unreachable from @start but is the head "
                                 "of a cycle; parser-unroll will not process it");
                         });
    }

    return result;
}

using PredMap = llvm::DenseMap<P4HIR::ParserStateOp,
                               llvm::SmallVector<P4HIR::ParserStateOp>>;
using AccessMap = llvm::DenseMap<P4HIR::ParserStateOp,
                                 llvm::SmallVector<StackAccess>>;
using BackEdge = std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>;

static PredMap buildPredMap(P4HIR::ParserOp parser) {
    PredMap preds;
    for (auto s : parser.states())
        for (auto next : s.getNextStates()) preds[next].push_back(s);
    return preds;
}

static llvm::DenseSet<P4HIR::ParserStateOp>
collectSCC(P4HIR::ParserStateOp loopHead, const PredMap &preds) {
    llvm::DenseSet<P4HIR::ParserStateOp> fwd;
    {
        llvm::SmallVector<P4HIR::ParserStateOp> wl = {loopHead};
        while (!wl.empty()) {
            auto s = wl.pop_back_val();
            if (!fwd.insert(s).second) continue;
            if (!isTerminal(s))
                for (auto n : s.getNextStates()) wl.push_back(n);
        }
    }
    llvm::DenseSet<P4HIR::ParserStateOp> bwd;
    {
        llvm::SmallVector<P4HIR::ParserStateOp> wl = {loopHead};
        while (!wl.empty()) {
            auto s = wl.pop_back_val();
            if (!bwd.insert(s).second) continue;
            if (auto it = preds.find(s); it != preds.end())
                for (auto p : it->second)
                    if (fwd.contains(p)) wl.push_back(p);
        }
    }
    llvm::DenseSet<P4HIR::ParserStateOp> scc;
    for (auto s : fwd)
        if (bwd.contains(s) && !isTerminal(s))
            scc.insert(s);
    return scc;
}

struct SCCInfo {
    llvm::DenseSet<P4HIR::ParserStateOp> members;
    // Combined {HSp} per SCC, keyed by loop head.
    llvm::DenseMap<P4HIR::ParserStateOp,
                   llvm::SmallVector<StackAccess>> combinedByHead;
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> headOf;
    // Per-state stacks that affect dedup of (state, M).
    llvm::DenseMap<P4HIR::ParserStateOp,
                   llvm::SmallVector<StackAccess>> relevantStacks;

    bool empty() const { return members.empty(); }
};

static SCCInfo buildSCCInfo(P4HIR::ParserOp parser,
                            llvm::ArrayRef<BackEdge> backEdges,
                            const AccessMap &stateAccesses,
                            const PredMap &predMap,
                            const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable) {
    SCCInfo scc;
    llvm::DenseSet<P4HIR::ParserStateOp> processedHeads;

    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declPos;
    {
        unsigned pos = 0;
        for (auto s : parser.states()) declPos[s] = pos++;
    }
    struct PendingSCC {
        P4HIR::ParserStateOp head;
        llvm::DenseSet<P4HIR::ParserStateOp> set;
    };
    llvm::SmallVector<PendingSCC> pending;
    {
        llvm::DenseSet<P4HIR::ParserStateOp> seenHead;
        for (auto &backEdge : backEdges) {
            auto head = backEdge.second;
            if (!seenHead.insert(head).second) continue;
            pending.push_back({head, collectSCC(head, predMap)});
        }
    }
    llvm::sort(pending, [&](const PendingSCC &a, const PendingSCC &b) {
        if (a.set.size() != b.set.size()) return a.set.size() < b.set.size();
        return declPos[a.head] < declPos[b.head];
    });

    for (auto &p : pending) {
        P4HIR::ParserStateOp loopHead = p.head;
        if (!processedHeads.insert(loopHead).second) continue;

        const auto &sccSet = p.set;

        bool anyUntrackable = false;
        for (auto s : sccSet)
            if (untrackable.contains(s)) { anyUntrackable = true; break; }
        if (anyUntrackable) continue;

        llvm::SmallVector<StackAccess> combined;
        {
            llvm::StringMap<size_t> seenSizes;
            llvm::StringSet<> warnedKeys;
            for (auto s : parser.states()) {
                if (!sccSet.contains(s)) continue;
                auto accIt = stateAccesses.find(s);
                assert(accIt != stateAccesses.end() &&
                       "SCC state missing from stateAccesses");
                for (auto &acc : accIt->second) {
                    auto [it, inserted] = seenSizes.insert({acc.key, acc.size});
                    if (inserted) {
                        combined.push_back(acc);
                    } else if (it->second != acc.size &&
                               warnedKeys.insert(acc.key).second) {
                        mlir::emitWarning(loopHead.getLoc(),
                            "header stack '" + acc.key +
                            "' appears with conflicting sizes in the same SCC; "
                            "unroll depth may be incorrect");
                    }
                }
            }
        }

        if (combined.empty()) {
            mlir::emitWarning(loopHead.getLoc(),
                "parser loop at state '" + loopHead.getName().str() +
                "' has no header stack operations; cannot infer unroll depth");
            continue;
        }

        size_t minSize = std::numeric_limits<size_t>::max();
        for (auto &acc : combined) minSize = std::min(minSize, acc.size);
        if (minSize > kDefaultMaxUnrollDepth) {
            mlir::emitWarning(loopHead.getLoc(),
                "parser loop at state '" + loopHead.getName().str() +
                "' would unroll to depth " + std::to_string(minSize) +
                " (> " + std::to_string(kDefaultMaxUnrollDepth) +
                "); skipping. Reduce header stack size or raise the limit.");
            continue;
        }

        bool hasOverlap = false;
        for (auto s : sccSet) {
            if (scc.headOf.count(s)) {
                hasOverlap = true;
                break;
            }
        }
        if (hasOverlap) {
            mlir::emitWarning(loopHead.getLoc(),
                "parser loop at state '" + loopHead.getName().str() +
                "' overlaps with a nested loop; outer loop will not be unrolled");
            continue;
        }

        scc.combinedByHead[loopHead] = std::move(combined);

        for (auto s : sccSet) {
            scc.members.insert(s);
            scc.headOf[s] = loopHead;
        }
    }

    {
        auto reachableHeads = [&](P4HIR::ParserStateOp s) {
            llvm::SmallVector<P4HIR::ParserStateOp> heads;
            llvm::DenseSet<P4HIR::ParserStateOp> seenHeads;
            llvm::DenseSet<P4HIR::ParserStateOp> visited;
            llvm::SmallVector<P4HIR::ParserStateOp> wl{s};
            while (!wl.empty()) {
                auto cur = wl.pop_back_val();
                if (!visited.insert(cur).second) continue;
                if (isTerminal(cur)) continue;
                if (auto it = scc.headOf.find(cur); it != scc.headOf.end()) {
                    if (seenHeads.insert(it->second).second)
                        heads.push_back(it->second);
                }
                for (auto next : cur.getNextStates()) wl.push_back(next);
            }
            return heads;
        };

        for (auto s : parser.states()) {
            if (isTerminal(s)) continue;
            llvm::SmallVector<StackAccess> rel;
            if (auto it = scc.headOf.find(s); it != scc.headOf.end()) {
                rel = scc.combinedByHead[it->second];
            } else {
                llvm::StringSet<> seenKeys;
                for (auto head : reachableHeads(s)) {
                    auto combIt = scc.combinedByHead.find(head);
                    if (combIt == scc.combinedByHead.end()) continue;
                    for (auto &acc : combIt->second)
                        if (seenKeys.insert(acc.key).second)
                            rel.push_back(acc);
                }
            }
            scc.relevantStacks[s] = std::move(rel);
        }
    }

    return scc;
}

// Definition 4: triple (state, call-number, M) stored for every visited node.
struct SymState {
    P4HIR::ParserStateOp state;
    unsigned callIndex;  // ind(state, M); 0 keeps original name
    IndexMap M;
};

struct SymResult {
    // Discovered (state, ind, M) triples (Definition 4).
    llvm::SmallVector<SymState> states;
    // (state, M) → ind, or nullopt when OOB (Stage 2, step 2).
    std::map<VisitedKey, std::optional<unsigned>> visitedMap;
    // Per-state {HSp}, used during M advancement (Stage 4, step 1).
    AccessMap accesses;

    struct SuccLookup {
        bool found = false;
        std::optional<unsigned> index;
    };
    SuccLookup lookupSuccessor(mlir::StringAttr name, const IndexMap &M) const {
        auto it = visitedMap.find({name, M});
        if (it == visitedMap.end()) return {};
        return {true, it->second};
    }
};

static SymResult runSymbolicExecution(P4HIR::ParserOp parser,
                                      const SCCInfo &scc,
                                      AccessMap stateAccesses) {
    SymResult result;
    result.accesses = std::move(stateAccesses);

    // Stage 1: initialisation.
    auto startState = parser.getStartState();
    if (!startState) return result;

    // WorkItem = (current state Snew, map M) — triple from Definition 4.
    struct WorkItem {
        P4HIR::ParserStateOp state;
        IndexMap M;
    };

    // ind counter per state name.
    llvm::StringMap<unsigned> callsCount;

    // BFS worklist seeded with (start, M=∅).
    std::deque<WorkItem> worklist;
    worklist.push_back({startState, {}});

    unsigned bfsIterations = 0;

    while (!worklist.empty()) {
        if (++bfsIterations > kMaxBFSStateInstances) {
            mlir::emitWarning(parser.getLoc(),
                "parser-unroll: BFS exceeded " +
                std::to_string(kMaxBFSStateInstances) +
                " state instances; aborting unroll for parser '" +
                parser.getSymName().str() + "'");
            result.states.clear();
            result.visitedMap.clear();
            return result;
        }

        auto [state, M] = std::move(worklist.front());
        worklist.pop_front();

        if (isTerminal(state)) continue;

        auto relIt = scc.relevantStacks.find(state);
        llvm::ArrayRef<StackAccess> relevant =
            (relIt != scc.relevantStacks.end())
                ? llvm::ArrayRef<StackAccess>(relIt->second)
                : llvm::ArrayRef<StackAccess>{};
        IndexMap restrictedM = M.restrictTo(relevant);
        VisitedKey key{state.getSymNameAttr(), restrictedM};

        auto accIt = result.accesses.find(state);
        assert(accIt != result.accesses.end() &&
               "state missing from accesses map");
        bool oob = M.isOOBForAny(accIt->second);

        // OOB → record nullopt so materialization wires this transition to @reject.
        if (oob) {
            result.visitedMap.emplace(std::move(key), std::nullopt);
            continue;
        }

        // Stage 2, step 2 / Stage 3, step 1: visited-state check and insertion.
        unsigned &countRef = callsCount[state.getSymName()];
        auto [vit, inserted] = result.visitedMap.try_emplace(std::move(key), countRef);
        if (!inserted) continue;
        unsigned idx = countRef++;

        result.states.push_back({state, idx, M});
        LLVM_DEBUG(llvm::dbgs() << "  visit " << state.getSymName()
                                 << " idx=" << idx << "\n");

        // Stage 4, step 1: advance M for successors using the state's own accesses.
        IndexMap M_after = M.advanced(accIt->second);

        for (auto succ : state.getNextStates()) {
            if (isTerminal(succ)) continue;
            worklist.push_back({succ, M_after});
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "  symbolic execution: "
                             << result.states.size() << " state instances, "
                             << result.visitedMap.size() << " visited keys\n");
    return result;
}

// Definition 3: ind(state, M) → name suffix; ind=0 keeps the original name.
static std::string stateName(llvm::StringRef base, unsigned idx) {
    if (idx == 0) return base.str();
    return (base + "_" + llvm::Twine(idx)).str();
}

// Phase 1 — Stage 4, step 1: clone each SymState with ind > 0.
static LogicalResult createClones(P4HIR::ParserOp parser,
                                   SymResult &sym,
                                   const SCCInfo &scc) {
    llvm::StringSet<> existingNames;
    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declPos;
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> insertCursor;

    unsigned pos = 0;
    for (auto s : parser.states()) {
        existingNames.insert(s.getSymName());
        declPos[s] = pos++;
        if (scc.members.contains(s)) {
            auto headIt = scc.headOf.find(s);
            assert(headIt != scc.headOf.end() &&
                   "SCC member missing from headOf — buildSCCInfo invariant");
            insertCursor[headIt->second] = s;
        }
    }

    struct CloneTask {
        unsigned callIndex;
        unsigned origDeclPos;
        P4HIR::ParserStateOp origState;
        P4HIR::ParserStateOp bucketKey;
        std::string cloneName;
    };
    llvm::SmallVector<CloneTask> tasks;
    for (auto &ss : sym.states) {
        if (ss.callIndex == 0) continue;
        P4HIR::ParserStateOp origState = ss.state;
        std::string name = stateName(origState.getSymName(), ss.callIndex);
        if (existingNames.count(name))
            return parser.emitError("parser-unroll: generated clone name '")
                   << name << "' collides with an existing parser state; "
                   << "rename the state to avoid the '_N' suffix pattern";
        existingNames.insert(name);
        auto declIt = declPos.find(origState);
        assert(declIt != declPos.end() &&
               "cloned state missing from declPos — should be in parser.states()");
        P4HIR::ParserStateOp bucketKey;
        if (auto headIt = scc.headOf.find(origState); headIt != scc.headOf.end()) {
            bucketKey = headIt->second;
        } else {
            bucketKey = origState;
            insertCursor.try_emplace(bucketKey, origState);
        }
        tasks.push_back({ss.callIndex, declIt->second,
                         origState, bucketKey, std::move(name)});
    }
    llvm::sort(tasks, [](CloneTask &a, CloneTask &b) {
        if (a.callIndex != b.callIndex) return a.callIndex < b.callIndex;
        if (a.origDeclPos != b.origDeclPos) return a.origDeclPos < b.origDeclPos;
        return a.origState.getOperation() < b.origState.getOperation();
    });

    mlir::OpBuilder builder(parser.getContext());

    for (auto &task : tasks) {
        auto cursorIt = insertCursor.find(task.bucketKey);
        assert(cursorIt != insertCursor.end() &&
               "bucket key missing from insertCursor map");

        builder.setInsertionPointAfter(cursorIt->second.getOperation());
        auto clone = mlir::cast<P4HIR::ParserStateOp>(
            builder.clone(*task.origState.getOperation()));
        clone.setSymName(task.cloneName);
        cursorIt->second = clone;
    }
    return success();
}

// Phase 2 — Stage 4, steps 2–3: redirect each transition to the ind-th clone,
// or to @reject for OOB successors.
static LogicalResult rewriteTransitions(P4HIR::ParserOp parser,
                                         SymResult &sym,
                                         const SCCInfo &scc,
                                         mlir::SymbolRefAttr rejectRef) {
    auto *ctx = parser.getContext();

    llvm::StringMap<P4HIR::ParserStateOp> stateByName;
    for (auto s : parser.states())
        stateByName[s.getSymName()] = s;

    struct StatePlan {
        P4HIR::ParserStateOp stateOp;
        llvm::StringMap<mlir::SymbolRefAttr> rewrites;
    };
    llvm::SmallVector<StatePlan, 16> plans;
    plans.reserve(sym.states.size());

    for (auto &ss : sym.states) {
        auto stateIt = stateByName.find(stateName(ss.state.getSymName(), ss.callIndex));
        if (stateIt == stateByName.end())
            return parser.emitError("parser-unroll: internal error — state '")
                   << stateName(ss.state.getSymName(), ss.callIndex)
                   << "' missing after createClones; this is a bug in the pass";
        P4HIR::ParserStateOp stateOp = stateIt->second;

        auto ssAccIt = sym.accesses.find(ss.state);
        assert(ssAccIt != sym.accesses.end() &&
               "sym state missing from accesses map");
        IndexMap M_after = ss.M.advanced(ssAccIt->second);

        StatePlan plan{stateOp, {}};
        for (auto succ : llvm::to_vector(stateOp.getNextStates())) {
            if (isTerminal(succ)) continue;

            mlir::StringAttr succNameAttr = succ.getSymNameAttr();
            llvm::StringRef succName = succNameAttr.getValue();
            if (plan.rewrites.count(succName)) continue;

            IndexMap lookupM;
            if (auto relIt = scc.relevantStacks.find(succ);
                relIt != scc.relevantStacks.end())
                lookupM = M_after.restrictTo(relIt->second);

            auto [found, succIdx] = sym.lookupSuccessor(succNameAttr, lookupM);
            if (!found)
                return stateOp.emitError(
                           "parser-unroll: BFS invariant violated — successor '")
                       << succName << "' not found in visited map";
            // succIdx: nullopt → OOB, wire to @reject
            if (succIdx && *succIdx == 0) continue;

            plan.rewrites[succName] = succIdx
                ? makeStateRef(ctx, stateName(succName, *succIdx))
                : rejectRef;
        }
        if (!plan.rewrites.empty())
            plans.push_back(std::move(plan));
    }

    for (auto &plan : plans)
        applyTransitionRewrites(plan.stateOp, plan.rewrites);
    return success();
}

static LogicalResult materializeUnrolled(P4HIR::ParserOp parser,
                                          SymResult &sym,
                                          const SCCInfo &scc) {
    if (sym.states.empty()) return success();

    mlir::SymbolRefAttr rejectRef;
    for (auto s : parser.states())
        if (s.isReject()) { rejectRef = s.getSymbolRef(); break; }
    if (!rejectRef)
        return parser.emitError("parser loop unrolling requires a @reject state");

    if (failed(createClones(parser, sym, scc)))
        return failure();
    return rewriteTransitions(parser, sym, scc, rejectRef);
}

struct ParserUnroll : public impl::ParserUnrollBase<ParserUnroll> {
    void runOnOperation() override {
        getOperation()->walk([&](P4HIR::ParserOp parser) {
            LLVM_DEBUG(llvm::dbgs() << "\n=== Parser Unroll: "
                                    << parser.getName() << " ===\n");

            auto backEdges = findBackEdges(parser);
            LLVM_DEBUG(llvm::dbgs() << "  back edges found: "
                                     << backEdges.size() << "\n");
            if (backEdges.empty()) return;

            auto predMap = buildPredMap(parser);

            // Collect per-state {HSp}.
            AccessMap stateAccesses;
            llvm::DenseSet<P4HIR::ParserStateOp> untrackable;
            for (auto s : parser.states()) {
                auto accs = computeStackAccesses(s);
                if (!accs) {
                    untrackable.insert(s);
                    stateAccesses[s] = {};
                } else {
                    stateAccesses[s] = std::move(*accs);
                }
            }

            auto scc = buildSCCInfo(parser, backEdges, stateAccesses, predMap,
                                     untrackable);
            LLVM_DEBUG(llvm::dbgs() << "  SCC members: " << scc.members.size()
                                     << " across " << scc.combinedByHead.size()
                                     << " loop(s)\n");
            if (scc.empty()) return;

            auto sym = runSymbolicExecution(parser, scc, std::move(stateAccesses));
            if (failed(materializeUnrolled(parser, sym, scc)))
                signalPassFailure();
        });
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createParserUnrollPass() {
    return std::make_unique<ParserUnroll>();
}

}  // namespace P4::P4MLIR
