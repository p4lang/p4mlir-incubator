// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <deque>
#include <limits>
#include <optional>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/ParserGraph.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-parser-unroll"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_PARSERUNROLL
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

static constexpr unsigned kDefaultMaxUnrollDepth = 64;
static constexpr unsigned kMaxBFSStateInstances = 100000;

// Header-stack storage identity, stored as DenseI64ArrayAttr for direct access.
using StackId = mlir::Attribute;

// Build a stack id.
static StackId makeStackId(mlir::Builder &attrBuilder, int64_t base,
                           llvm::ArrayRef<int64_t> path) {
    llvm::SmallVector<int64_t, 5> elements;
    elements.push_back(base);
    elements.append(path.begin(), path.end());
    return attrBuilder.getDenseI64ArrayAttr(elements);
}

// Order two stack ids.
static bool stackIdLess(StackId lhs, StackId rhs) {
    auto lhsArr = mlir::cast<mlir::DenseI64ArrayAttr>(lhs);
    auto rhsArr = mlir::cast<mlir::DenseI64ArrayAttr>(rhs);
    for (size_t i = 0, n = std::min(lhsArr.size(), rhsArr.size()); i < n; ++i) {
        if (lhsArr[i] != rhsArr[i]) return lhsArr[i] < rhsArr[i];
    }
    return lhsArr.size() < rhsArr.size();
}

// Value-numbers each stack base deterministically, on first encounter.
// Named variables use name-based keys so cloned states (which share the name
// but have distinct SSA values) map to the same StackId as the original.
struct StackNumbering {
    llvm::StringMap<unsigned> nameIds;
    llvm::DenseMap<mlir::Value, unsigned> valueIds;
    unsigned nextId = 0;
    unsigned forName(llvm::StringRef name) {
        auto [it, inserted] = nameIds.try_emplace(name, nextId);
        if (inserted) ++nextId;
        return it->second;
    }
    unsigned forValue(mlir::Value base) {
        auto [it, inserted] = valueIds.try_emplace(base, nextId);
        if (inserted) ++nextId;
        return it->second;
    }
};

static std::string renderStackId(StackId id) {
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    for (auto [index, field] :
         llvm::enumerate(mlir::cast<mlir::DenseI64ArrayAttr>(id).asArrayRef())) {
        stream << (index == 0 ? "#" : ".") << field;
    }
    return rendered;
}

using ValueMap = llvm::DenseMap<StackId, mlir::TypedAttr>;

struct StackAccess {
    StackId key;         // the stack variable (in {HSp})
    size_t size;         // OOB when the highest index used reaches size
    unsigned count = 1;  // number of .next accesses to this stack in the state
};

class IndexMap {
 public:
    bool isOOBFor(const StackAccess &access) const {
        auto it = data_.find(access.key);
        // Accesses use indices M[key] .. M[key]+count-1; OOB if the last reaches size.
        return it != data_.end() && it->second + access.count > access.size;
    }
    bool isOOBForAny(llvm::ArrayRef<StackAccess> accesses) const {
        for (auto &access : accesses)
            if (isOOBFor(access)) return true;
        return false;
    }
    // IndexMap advanced past these accesses.
    IndexMap advanced(llvm::ArrayRef<StackAccess> accesses) const {
        IndexMap result = *this;
        for (auto &access : accesses) result.data_[access.key] += access.count;
        return result;
    }
    // IndexMap restricted to relevant stacks.
    IndexMap restrictTo(llvm::ArrayRef<StackAccess> relevant) const {
        IndexMap result;
        for (auto &access : relevant) {
            auto it = data_.find(access.key);
            if (it != data_.end()) result.data_[access.key] = it->second;
        }
        return result;
    }

    unsigned indexOf(StackId key) const {
        auto it = data_.find(key);
        return it == data_.end() ? 0 : it->second;
    }
    // Encode as sorted [stackId, index] ArrayAttr.
    mlir::ArrayAttr encode(mlir::MLIRContext *context) const {
        llvm::SmallVector<std::pair<StackId, unsigned>> entries(data_.begin(), data_.end());
        llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
            return stackIdLess(lhs.first, rhs.first);
        });
        mlir::Builder attrBuilder(context);
        llvm::SmallVector<mlir::Attribute> encoded;
        for (auto &entry : entries)
            encoded.push_back(
                mlir::ArrayAttr::get(context, {entry.first, attrBuilder.getIndexAttr(entry.second)}));
        return mlir::ArrayAttr::get(context, encoded);
    }

 private:
    llvm::DenseMap<StackId, unsigned> data_;
};

// Encode value map as sorted [stackId, value] ArrayAttr.
static mlir::ArrayAttr encodeValueMap(mlir::MLIRContext *context, const ValueMap &valueMap) {
    llvm::SmallVector<std::pair<StackId, mlir::TypedAttr>> entries(valueMap.begin(),
                                                                   valueMap.end());
    llvm::sort(entries,
               [](const auto &lhs, const auto &rhs) { return stackIdLess(lhs.first, rhs.first); });
    llvm::SmallVector<mlir::Attribute> encoded;
    for (auto &entry : entries)
        encoded.push_back(mlir::ArrayAttr::get(context, {entry.first, entry.second}));
    return mlir::ArrayAttr::get(context, encoded);
}

// Make a stack positin key as an Attribute.
static mlir::Attribute makeVisitedKey(mlir::MLIRContext *context, mlir::StringAttr name,
                                      const IndexMap &indexMap, const ValueMap &valueMap) {
    return mlir::ArrayAttr::get(
        context, {name, indexMap.encode(context), encodeValueMap(context, valueMap)});
}

// Header-stack element count for a (reference) type, if it is a stack.
static mlir::FailureOr<size_t> stackSizeOf(mlir::Type type) {
    if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type)) type = ref.getObjectType();
    if (auto stackType = mlir::dyn_cast<P4HIR::HeaderStackType>(type))
        return stackType.getArraySize();
    return mlir::failure();
}

// Stack variable id that a value refers to, if any.
static mlir::FailureOr<StackId> getStackId(mlir::Value value, StackNumbering &numbering) {
    mlir::Builder attrBuilder(value.getContext());
    llvm::SmallVector<int64_t, 4> reversePath;
    int64_t base = 0;

    while (true) {
        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            base = numbering.forValue(arg);
            break;
        }
        auto *definingOp = value.getDefiningOp();
        if (!definingOp) return mlir::failure();

        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(definingOp)) {
            if (auto name = var.getName())
                base = numbering.forName(*name);
            else
                base = numbering.forValue(var.getResult());
            break;
        }
        if (auto fieldRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(definingOp)) {
            reversePath.push_back(fieldRef.getFieldIndex());
            value = fieldRef.getInput();
            continue;
        }
        if (auto structExtract = mlir::dyn_cast<P4HIR::StructExtractOp>(definingOp)) {
            reversePath.push_back(structExtract.getFieldIndex());
            value = structExtract.getInput();
            continue;
        }
        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(definingOp)) {
            value = readOp.getRef();
            continue;
        }
        return mlir::failure();
    }

    llvm::SmallVector<int64_t, 4> path(reversePath.rbegin(), reversePath.rend());
    return makeStackId(attrBuilder, base, path);
}

// Fold a value to constant attribute.
static mlir::TypedAttr evalConstAttr(mlir::Value value, const ValueMap &valueMap,
                                     StackNumbering &numbering) {
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) return {};

    if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(definingOp)) {
        if (auto key = getStackId(readOp.getRef(), numbering); succeeded(key)) {
            auto it = valueMap.find(*key);
            if (it != valueMap.end()) return it->second;
        }
        return {};
    }

    llvm::SmallVector<mlir::Attribute> operandConsts;
    for (mlir::Value operand : definingOp->getOperands()) {
        auto folded = evalConstAttr(operand, valueMap, numbering);
        if (!folded) return {};
        operandConsts.push_back(folded);
    }
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::failed(definingOp->fold(operandConsts, results)) || results.size() != 1) return {};
    if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(results[0]))
        return mlir::dyn_cast_if_present<mlir::TypedAttr>(attr);
    if (auto foldedValue = llvm::dyn_cast_if_present<mlir::Value>(results[0]))
        return evalConstAttr(foldedValue, valueMap, numbering);
    return {};
}

// Fold a value to constant integer.
static mlir::FailureOr<llvm::APSInt> evalConst(mlir::Value value, const ValueMap &valueMap,
                                               StackNumbering &numbering) {
    if (auto attribute = evalConstAttr(value, valueMap, numbering))
        if (auto constInt = P4HIR::getConstantInt(attribute))
            return *constInt;
    return mlir::failure();
}

// Symbolically interpret a state's body, updating the value map.
static ValueMap interpretState(
    P4HIR::ParserStateOp state, ValueMap valueMap,
    llvm::function_ref<void(P4HIR::ArrayElementRefOp, const ValueMap &)> onAccess,
    StackNumbering &numbering) {
    state.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op)) {
            onAccess(arrayElementRef, valueMap);
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            auto key = getStackId(assignOp.getRef(), numbering);
            if (failed(key)) return;
            if (auto attribute = evalConstAttr(assignOp.getValue(), valueMap, numbering))
                valueMap[*key] = attribute;
            else
                valueMap.erase(*key);
        }
    });
    return valueMap;
}

// Collect index-variable ids referenced by a value.
static void collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out,
                               StackNumbering &numbering) {
    if (auto readOp = value.getDefiningOp<P4HIR::ReadOp>()) {
        if (auto key = getStackId(readOp.getRef(), numbering); succeeded(key)) out.insert(*key);
        return;
    }
    if (auto *definingOp = value.getDefiningOp())
        for (mlir::Value operand : definingOp->getOperands())
            collectVarsInIndex(operand, out, numbering);
}

// Collect all index variables used across the parser.
static llvm::DenseSet<StackId> collectIndexVars(P4HIR::ParserOp parser, StackNumbering &numbering) {
    llvm::DenseSet<StackId> out;
    parser.walk([&](P4HIR::ArrayElementRefOp arrayElementRef) {
        collectVarsInIndex(arrayElementRef.getIndex(), out, numbering);
    });
    return out;
}

// Restrict a value map to the given keys.
static ValueMap restrictValueMap(const ValueMap &valueMap, const llvm::DenseSet<StackId> &keep) {
    ValueMap restricted;
    for (auto &entry : valueMap)
        if (keep.count(entry.first)) restricted.insert(entry);
    return restricted;
}

// Definition 5 / ParserStructure: builds {HSp} for a single state.
static mlir::FailureOr<llvm::SmallVector<StackAccess>> computeStackAccesses(
    P4HIR::ParserStateOp state, StackNumbering &numbering) {
    llvm::SmallVector<StackAccess> result;
    llvm::DenseSet<StackId> seen;
    llvm::DenseMap<StackId, unsigned> counts;
    bool unidentified = false;

    auto record = [&](mlir::Value input) {
        auto size = stackSizeOf(input.getType());
        if (failed(size) || *size == 0) return;
        auto key = getStackId(input, numbering);
        if (failed(key)) {
            unidentified = true;
            return;
        }
        if (!seen.insert(*key).second) return;
        result.push_back({std::move(*key), *size});
    };

    state.walk([&](mlir::Operation *op) {
        if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op)) {
            mlir::Value idx = elementRef.getIndex();
            if (evalConstAttr(idx, ValueMap{}, numbering)) return;

            auto *arrayDef = elementRef.getInput().getDefiningOp();
            if (!arrayDef) return;
            auto dataRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(arrayDef);
            if (!dataRef || dataRef.getFieldName() != "data") return;

            // Count only true .next (nextIndex-derived) accesses - those are the
            // ones that advance nextIndex. Explicit indices (stack[expr]) record
            // the stack but do not add to the per-state increment count.
            bool isNext = false;
            if (auto readOp = idx.getDefiningOp<P4HIR::ReadOp>()) {
                if (auto nextIndexRef = readOp.getRef().getDefiningOp<P4HIR::StructFieldRefOp>())
                    isNext = nextIndexRef.getFieldName() == "nextIndex";
            } else if (auto structExtract = idx.getDefiningOp<P4HIR::StructExtractOp>()) {
                isNext = structExtract.getFieldName() == "nextIndex";
            }
            if (isNext)
                if (auto key = getStackId(dataRef.getInput(), numbering); succeeded(key))
                    ++counts[*key];
            record(dataRef.getInput());
        }
    });

    if (unidentified) {
        state.emitWarning()
            << "cannot determine identity of header stack accessed in state '"
            << state.getName() << "'; not unrolling this loop";
        return mlir::failure();
    }
    for (auto &access : result) {
        auto it = counts.find(access.key);
        access.count = std::max<unsigned>(1, it == counts.end() ? 0 : it->second);
    }
    return result;
}

// Apply state-name rewrites to a state's transitions and select cases.
static void applyTransitionRewrites(P4HIR::ParserStateOp state,
                                    const llvm::StringMap<mlir::SymbolRefAttr> &rewrites) {
    if (rewrites.empty()) return;
    auto matches = [&](mlir::SymbolRefAttr ref) -> mlir::SymbolRefAttr {
        auto it = rewrites.find(ref.getLeafReference().getValue());
        if (it == rewrites.end()) return {};
        return it->second;
    };
    state.walk([&](mlir::Operation *op) {
        if (auto transition = mlir::dyn_cast<P4HIR::ParserTransitionOp>(op)) {
            if (auto replacement = matches(transition.getStateAttr()))
                transition.setStateAttr(replacement);
        } else if (auto selectCase = mlir::dyn_cast<P4HIR::ParserSelectCaseOp>(op)) {
            if (auto replacement = matches(selectCase.getStateAttr()))
                selectCase.setStateAttr(replacement);
        }
    });
}

struct DfsFrame {
    P4HIR::ParserStateOp state;
    llvm::SmallVector<P4HIR::ParserStateOp> nexts;
    unsigned nextIdx = 0;
};

// DFS from a seed, invoking the callback on each back edge.
template <typename OnBackEdge>
static void dfsFindBackEdges(P4HIR::ParserStateOp seed,
                             llvm::DenseSet<P4HIR::ParserStateOp> &visited, OnBackEdge onBackEdge) {
    if (visited.contains(seed)) return;
    llvm::DenseSet<P4HIR::ParserStateOp> onStack;
    llvm::SmallVector<DfsFrame> stack;

    visited.insert(seed);
    onStack.insert(seed);
    stack.push_back({seed, llvm::SmallVector<P4HIR::ParserStateOp>(seed.getNextStates()), 0});

    while (!stack.empty()) {
        bool descended = false;
        while (stack.back().nextIdx < stack.back().nexts.size()) {
            P4HIR::ParserStateOp next = stack.back().nexts[stack.back().nextIdx++];
            if (onStack.contains(next)) {
                onBackEdge(stack.back().state, next);
            } else if (!visited.contains(next)) {
                visited.insert(next);
                onStack.insert(next);
                stack.push_back(
                    {next, llvm::SmallVector<P4HIR::ParserStateOp>(next.getNextStates()), 0});
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
static llvm::SmallVector<std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>> findBackEdges(
    P4HIR::ParserOp parser) {
    llvm::SmallVector<std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>> result;

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

using AccessMap = llvm::DenseMap<P4HIR::ParserStateOp, llvm::SmallVector<StackAccess>>;
using BackEdge = std::pair<P4HIR::ParserStateOp, P4HIR::ParserStateOp>;

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

struct SCCInfo {
    llvm::DenseSet<P4HIR::ParserStateOp> members;
    // Combined {HSp} per SCC, keyed by loop head.
    llvm::DenseMap<P4HIR::ParserStateOp, llvm::SmallVector<StackAccess>> combinedByHead;
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> headOf;
    // Per-state stacks that affect dedup of (state, M).
    llvm::DenseMap<P4HIR::ParserStateOp, llvm::SmallVector<StackAccess>> relevantStacks;

    bool empty() const { return members.empty(); }
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

// Merge one access into a combined set.
static void addCombinedAccess(const StackAccess &access, P4HIR::ParserStateOp loopHead,
                              llvm::SmallVector<StackAccess> &combined,
                              llvm::DenseMap<StackId, size_t> &seenSizes,
                              llvm::DenseSet<StackId> &warnedKeys) {
    auto [it, inserted] = seenSizes.insert({access.key, access.size});
    if (inserted) {
        combined.push_back(access);
        return;
    }
    if (it->second != access.size && warnedKeys.insert(access.key).second)
        loopHead.emitWarning()
            << "header stack '" << renderStackId(access.key)
            << "' appears with conflicting sizes in the same SCC; "
               "unroll depth may be incorrect";
}

// Combined {HSp} across an SCC's states.
static llvm::SmallVector<StackAccess> combineSCCAccesses(
    P4HIR::ParserOp parser, P4HIR::ParserStateOp loopHead,
    const llvm::DenseSet<P4HIR::ParserStateOp> &sccSet, const AccessMap &stateAccesses) {
    llvm::SmallVector<StackAccess> combined;
    llvm::DenseMap<StackId, size_t> seenSizes;
    llvm::DenseSet<StackId> warnedKeys;
    for (auto stateOp : parser.states()) {
        if (!sccSet.contains(stateOp)) continue;
        auto accessIt = stateAccesses.find(stateOp);
        assert(accessIt != stateAccesses.end() && "SCC state missing from stateAccesses");
        for (auto &access : accessIt->second)
            addCombinedAccess(access, loopHead, combined, seenSizes, warnedKeys);
    }
    return combined;
}

// Accept one SCC as an unrollable loop.
static void acceptLoopSCC(SCCInfo &scc, const PendingSCC &candidate, P4HIR::ParserOp parser,
                          const AccessMap &stateAccesses,
                          const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable) {
    P4HIR::ParserStateOp loopHead = candidate.head;
    const auto &sccSet = candidate.members;

    for (auto stateOp : sccSet)
        if (untrackable.contains(stateOp)) return;

    auto combined = combineSCCAccesses(parser, loopHead, sccSet, stateAccesses);
    if (combined.empty()) {
        loopHead.emitWarning()
            << "parser loop at state '" << loopHead.getName()
            << "' has no header stack operations; cannot infer unroll depth";
        return;
    }

    size_t minSize = std::numeric_limits<size_t>::max();
    for (auto &access : combined) minSize = std::min(minSize, access.size);
    if (minSize > kDefaultMaxUnrollDepth) {
        loopHead.emitWarning()
            << "parser loop at state '" << loopHead.getName()
            << "' would unroll to depth " << minSize << " (> "
            << kDefaultMaxUnrollDepth
            << "); skipping. Reduce header stack size or raise the limit.";
        return;
    }

    scc.combinedByHead[loopHead] = std::move(combined);
    for (auto stateOp : sccSet) {
        scc.members.insert(stateOp);
        scc.headOf[stateOp] = loopHead;
    }
}

// Unique stack accesses reachable from start via DFS.
// accessesFor returns the accesses associated with each visited state.
template <typename AccessProvider>
static llvm::SmallVector<StackAccess> reachable(P4HIR::ParserStateOp start,
                                                AccessProvider accessesFor) {
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
static void computeRelevantStacks(P4HIR::ParserOp parser, SCCInfo &scc,
                                  const AccessMap &stateAccesses) {
    bool acyclic = scc.combinedByHead.empty();
    for (auto stateOp : parser.states()) {
        if (stateOp.isTerminal()) continue;
        llvm::SmallVector<StackAccess> relevant;
        if (acyclic) {
            relevant = reachable(stateOp,
                                [&](P4HIR::ParserStateOp state) -> llvm::ArrayRef<StackAccess> {
                                    auto it = stateAccesses.find(state);
                                    if (it == stateAccesses.end()) return {};
                                    return it->second;
                                });
        } else if (auto it = scc.headOf.find(stateOp); it != scc.headOf.end()) {
            relevant = scc.combinedByHead[it->second];
        } else {
            relevant = reachable(stateOp,
                                [&](P4HIR::ParserStateOp state) -> llvm::ArrayRef<StackAccess> {
                                    auto headIt = scc.headOf.find(state);
                                    if (headIt == scc.headOf.end()) return {};
                                    auto combinedIt = scc.combinedByHead.find(headIt->second);
                                    if (combinedIt == scc.combinedByHead.end()) return {};
                                    return combinedIt->second;
                                });
        }
        scc.relevantStacks[stateOp] = std::move(relevant);
    }
}

// Build per-loop SCC info.
static SCCInfo buildSCCInfo(
    P4HIR::ParserOp parser, llvm::ArrayRef<BackEdge> backEdges,
    const AccessMap &stateAccesses,
    const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable,
    const llvm::DenseMap<P4HIR::ParserStateOp, unsigned> &declarationPos) {
    SCCInfo scc;
    auto pending = collectPendingSCCs(parser, backEdges, declarationPos);
    for (auto &candidate : pending)
        acceptLoopSCC(scc, candidate, parser, stateAccesses, untrackable);
    computeRelevantStacks(parser, scc, stateAccesses);
    return scc;
}

// Definition 4: triple (state, call-number, M) stored for every visited node.
struct SymbolicState {
    P4HIR::ParserStateOp state;
    unsigned callIndex;  // ind(state, M); 0 keeps original name
    IndexMap indexMap;
    ValueMap entryValueMap;
};

struct SymbolicResult {
    // Discovered (state, ind, M) triples (Definition 4).
    llvm::SmallVector<SymbolicState> states;
    // (state, M) -> ind, or nullopt when OOB (Stage 2, step 2).
    llvm::DenseMap<mlir::Attribute, std::optional<unsigned>> visitedMap;
    // Per-state {HSp}, used during M advancement (Stage 4, step 1).
    AccessMap accesses;
    llvm::DenseSet<StackId> indexVars;

    struct successorLookup {
        bool found = false;
        std::optional<unsigned> index;
    };
    successorLookup lookupsuccessor(mlir::StringAttr name, const IndexMap &indexMap,
                                    const ValueMap &valueMap) const {
        auto it = visitedMap.find(makeVisitedKey(name.getContext(), name, indexMap, valueMap));
        if (it == visitedMap.end()) return {};
        return {true, it->second};
    }
};

// BFS symbolic execution producing the (state, ind, M) instances to clone.
static SymbolicResult runSymbolicExecution(P4HIR::ParserOp parser, const SCCInfo &scc,
                                           AccessMap stateAccesses, StackNumbering &numbering) {
    SymbolicResult result;
    result.accesses = std::move(stateAccesses);
    result.indexVars = collectIndexVars(parser, numbering);

    // Stage 1: initialisation.
    auto startState = parser.getStartState();
    if (!startState) return result;

    // WorkItem = (current state Snew, map M) - triple from Definition 4.
    struct WorkItem {
        P4HIR::ParserStateOp state;
        IndexMap indexMap;
        ValueMap valueMap;
    };

    // ind counter per state name.
    llvm::StringMap<unsigned> callsCount;

    // BFS worklist seeded with (start, M={}).
    std::deque<WorkItem> worklist;
    worklist.push_back({startState, IndexMap(), ValueMap()});

    unsigned bfsIterations = 0;

    while (!worklist.empty()) {
        if (++bfsIterations > kMaxBFSStateInstances) {
            parser.emitWarning()
                << "parser-unroll: BFS exceeded " << kMaxBFSStateInstances
                << " state instances; aborting unroll for parser '"
                << parser.getSymName() << "'";
            result.states.clear();
            result.visitedMap.clear();
            return result;
        }

        auto [state, indexMap, valueMap] = std::move(worklist.front());
        worklist.pop_front();

        if (state.isTerminal()) continue;

        auto relIt = scc.relevantStacks.find(state);
        llvm::ArrayRef<StackAccess> relevant = (relIt != scc.relevantStacks.end())
                                                   ? llvm::ArrayRef<StackAccess>(relIt->second)
                                                   : llvm::ArrayRef<StackAccess>{};
        IndexMap restrictedIndexMap = indexMap.restrictTo(relevant);
        ValueMap restrictedValueMap = restrictValueMap(valueMap, result.indexVars);
        mlir::Attribute key = makeVisitedKey(parser.getContext(), state.getSymNameAttr(),
                                             restrictedIndexMap, restrictedValueMap);

        auto accessesIt = result.accesses.find(state);
        assert(accessesIt != result.accesses.end() && "state missing from accesses map");
        bool oob = indexMap.isOOBForAny(accessesIt->second);

        // OOB -> record nullopt so materialization wires this transition to @reject.
        if (oob) {
            result.visitedMap.try_emplace(key, std::nullopt);
            continue;
        }

        // Stage 2, step 2 / Stage 3, step 1: visited-state check and insertion.
        unsigned &countRef = callsCount[state.getSymName()];
        auto [visitedIt, inserted] = result.visitedMap.try_emplace(key, countRef);
        if (!inserted) continue;
        unsigned idx = countRef++;

        result.states.push_back({state, idx, indexMap, valueMap});
        LLVM_DEBUG(llvm::dbgs() << "  visit " << state.getSymName() << " idx=" << idx << "\n");

        // Stage 4, step 1: advance M for successors using the state's own accesses.
        IndexMap indexMapAfter = indexMap.advanced(accessesIt->second);
        ValueMap valueMapAfter = interpretState(
            state, valueMap, [](P4HIR::ArrayElementRefOp, const ValueMap &) {}, numbering);

        for (auto successor : state.getNextStates()) {
            if (successor.isTerminal()) continue;
            worklist.push_back({successor, indexMapAfter, valueMapAfter});
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "  symbolic execution: " << result.states.size()
                            << " state instances, " << result.visitedMap.size()
                            << " visited keys\n");
    return result;
}

// Definition 3: ind(state, M) -> name suffix; ind=0 keeps the original name.
static std::string stateName(llvm::StringRef base, unsigned idx) {
    if (idx == 0) return base.str();
    return (base + "_" + llvm::Twine(idx)).str();
}

// Phase 1 - Stage 4, step 1: clone each SymbolicState with ind > 0.
static LogicalResult createClones(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                  const SCCInfo &scc) {
    llvm::StringSet<> existingNames;
    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declarationPos;
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> insertCursor;

    unsigned pos = 0;
    for (auto stateOp : parser.states()) {
        existingNames.insert(stateOp.getSymName());
        declarationPos[stateOp] = pos++;
        if (scc.members.contains(stateOp)) {
            auto headIt = scc.headOf.find(stateOp);
            assert(headIt != scc.headOf.end() &&
                   "SCC member missing from headOf - buildSCCInfo invariant");
            insertCursor[headIt->second] = stateOp;
        }
    }

    struct CloneTask {
        unsigned callIndex;
        unsigned originalDeclarationPos;
        P4HIR::ParserStateOp originalState;
        P4HIR::ParserStateOp bucketKey;
        std::string cloneName;
    };
    llvm::SmallVector<CloneTask> tasks;
    for (auto &symbolicState : symbolicResult.states) {
        if (symbolicState.callIndex == 0) continue;
        P4HIR::ParserStateOp originalState = symbolicState.state;
        std::string name = stateName(originalState.getSymName(), symbolicState.callIndex);
        if (existingNames.count(name))
            return parser.emitError("parser-unroll: generated clone name '")
                   << name << "' collides with an existing parser state; "
                   << "rename the state to avoid the '_N' suffix pattern";
        existingNames.insert(name);
        auto declarationIt = declarationPos.find(originalState);
        assert(declarationIt != declarationPos.end() &&
               "cloned state missing from declarationPos - should be in parser.states()");
        P4HIR::ParserStateOp bucketKey;
        if (auto headIt = scc.headOf.find(originalState); headIt != scc.headOf.end()) {
            bucketKey = headIt->second;
        } else {
            bucketKey = originalState;
            insertCursor.try_emplace(bucketKey, originalState);
        }
        tasks.push_back({symbolicState.callIndex, declarationIt->second, originalState, bucketKey,
                         std::move(name)});
    }
    llvm::sort(tasks, [](CloneTask &lhs, CloneTask &rhs) {
        if (lhs.callIndex != rhs.callIndex) return lhs.callIndex < rhs.callIndex;
        if (lhs.originalDeclarationPos != rhs.originalDeclarationPos)
            return lhs.originalDeclarationPos < rhs.originalDeclarationPos;
        return lhs.originalState.getOperation() < rhs.originalState.getOperation();
    });

    mlir::OpBuilder builder(parser.getContext());

    for (auto &task : tasks) {
        auto cursorIt = insertCursor.find(task.bucketKey);
        assert(cursorIt != insertCursor.end() && "bucket key missing from insertCursor map");

        builder.setInsertionPointAfter(cursorIt->second.getOperation());
        auto clone =
            mlir::cast<P4HIR::ParserStateOp>(builder.clone(*task.originalState.getOperation()));
        clone.setSymName(task.cloneName);
        cursorIt->second = clone;
    }
    return success();
}

// Phase 2 - Stage 4, steps 2-3: redirect each transition to the ind-th clone,
// or to @reject for OOB successors.
static LogicalResult rewriteTransitions(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                        const SCCInfo &scc, mlir::SymbolRefAttr rejectRef,
                                        StackNumbering &numbering) {
    auto *context = parser.getContext();

    llvm::StringMap<P4HIR::ParserStateOp> stateByName;
    for (auto stateOp : parser.states()) stateByName[stateOp.getSymName()] = stateOp;

    struct StatePlan {
        P4HIR::ParserStateOp stateOp;
        llvm::StringMap<mlir::SymbolRefAttr> rewrites;
    };
    llvm::SmallVector<StatePlan, 16> plans;
    plans.reserve(symbolicResult.states.size());

    for (auto &symbolicState : symbolicResult.states) {
        auto stateIt =
            stateByName.find(stateName(symbolicState.state.getSymName(), symbolicState.callIndex));
        if (stateIt == stateByName.end())
            return parser.emitError("parser-unroll: internal error - state '")
                   << stateName(symbolicState.state.getSymName(), symbolicState.callIndex)
                   << "' missing after createClones; this is a bug in the pass";
        P4HIR::ParserStateOp stateOp = stateIt->second;

        auto symbolicStateAccessIt = symbolicResult.accesses.find(symbolicState.state);
        assert(symbolicStateAccessIt != symbolicResult.accesses.end() &&
               "symbolicResult state missing from accesses map");
        IndexMap indexMapAfter = symbolicState.indexMap.advanced(symbolicStateAccessIt->second);
        ValueMap lookupValueMap =
            restrictValueMap(interpretState(
                                 symbolicState.state, symbolicState.entryValueMap,
                                 [](P4HIR::ArrayElementRefOp, const ValueMap &) {}, numbering),
                             symbolicResult.indexVars);

        StatePlan plan{stateOp, {}};
        for (auto successor : llvm::to_vector(stateOp.getNextStates())) {
            if (successor.isTerminal()) continue;

            mlir::StringAttr successorNameAttr = successor.getSymNameAttr();
            llvm::StringRef successorName = successorNameAttr.getValue();
            if (plan.rewrites.count(successorName)) continue;

            IndexMap lookupIndexMap;
            if (auto relIt = scc.relevantStacks.find(successor); relIt != scc.relevantStacks.end())
                lookupIndexMap = indexMapAfter.restrictTo(relIt->second);

            auto [found, successorIdx] =
                symbolicResult.lookupsuccessor(successorNameAttr, lookupIndexMap, lookupValueMap);
            if (!found)
                return stateOp.emitError("parser-unroll: BFS invariant violated - successor '")
                       << successorName << "' not found in visited map";
            // successorIdx: nullopt -> OOB, wire to @reject
            if (successorIdx && *successorIdx == 0) continue;

            plan.rewrites[successorName] =
                successorIdx
                    ? mlir::FlatSymbolRefAttr::get(context, stateName(successorName, *successorIdx))
                    : rejectRef;
        }
        if (!plan.rewrites.empty()) plans.push_back(std::move(plan));
    }

    for (auto &plan : plans) applyTransitionRewrites(plan.stateOp, plan.rewrites);
    return success();
}

// Stage 4.1: substitute the concrete header-stack index (map M) into each
// materialised state, turning stack.next accesses into constant-index accesses.
static void substituteConstantIndices(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                      StackNumbering &numbering) {
    llvm::StringMap<P4HIR::ParserStateOp> stateByName;
    for (auto stateOp : parser.states()) stateByName[stateOp.getSymName()] = stateOp;

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
        auto stateIt =
            stateByName.find(stateName(symbolicState.state.getSymName(), symbolicState.callIndex));
        if (stateIt == stateByName.end()) continue;

        llvm::DenseMap<StackId, unsigned> count;
        if (auto accessIt = symbolicResult.accesses.find(symbolicState.state);
            accessIt != symbolicResult.accesses.end())
            for (auto &access : accessIt->second) count[access.key] = access.count;

        llvm::DenseMap<StackId, unsigned> occurrence;
        stateIt->second.walk([&](mlir::Operation *op) {
            mlir::Value idxVal;
            if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op))
                idxVal = arrayElementRef.getIndex();
            else if (auto arrayGet = mlir::dyn_cast<P4HIR::ArrayGetOp>(op))
                idxVal = arrayGet.getIndex();
            else
                return;

            std::optional<StackId> key;
            int64_t value = 0;
            if ((key = nextIndexKeyOf(idxVal))) {
                value =
                    static_cast<int64_t>(symbolicState.indexMap.indexOf(*key)) + occurrence[*key]++;
            } else if (auto binOp = idxVal.getDefiningOp<P4HIR::BinOp>();
                       binOp && binOp.getKind() == P4HIR::BinOpKind::Sub) {
                if (auto rhsConstOp = binOp.getRhs().getDefiningOp<P4HIR::ConstOp>())
                    if (auto rhsIntAttr = mlir::dyn_cast<P4HIR::IntAttr>(rhsConstOp.getValue());
                        rhsIntAttr && (key = nextIndexKeyOf(binOp.getLhs())))
                        value = static_cast<int64_t>(symbolicState.indexMap.indexOf(*key)) +
                                count[*key] - rhsIntAttr.getValue().getSExtValue();
            }
            if (!key || value < 0) return;

            auto idxType = mlir::dyn_cast<P4HIR::BitsType>(idxVal.getType());
            if (!idxType) return;

            builder.setInsertionPoint(op);
            auto constOp =
                P4HIR::ConstOp::create(builder, op->getLoc(), P4HIR::IntAttr::get(idxType, value));
            if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op))
                arrayElementRef.getIndexMutable().assign(constOp.getResult());
            else
                mlir::cast<P4HIR::ArrayGetOp>(op).getIndexMutable().assign(constOp.getResult());
        });
    }
}

// Fold non-constant stack indices that evaluate to a constant into ConstOps.
static void substituteExplicitIndices(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                      StackNumbering &numbering) {
    if (symbolicResult.indexVars.empty()) return;

    llvm::StringMap<P4HIR::ParserStateOp> stateByName;
    for (auto stateOp : parser.states()) stateByName[stateOp.getSymName()] = stateOp;

    mlir::OpBuilder builder(parser.getContext());
    for (auto &symbolicState : symbolicResult.states) {
        auto stateIt =
            stateByName.find(stateName(symbolicState.state.getSymName(), symbolicState.callIndex));
        if (stateIt == stateByName.end()) continue;

        interpretState(
            stateIt->second, symbolicState.entryValueMap,
            [&](P4HIR::ArrayElementRefOp elementRef, const ValueMap &valueMap) {
                mlir::Value idx = elementRef.getIndex();
                if (evalConstAttr(idx, ValueMap{}, numbering)) return;

                auto idxType = mlir::dyn_cast<P4HIR::BitsType>(idx.getType());
                if (!idxType) return;

                auto value = evalConst(idx, valueMap, numbering);
                if (failed(value) || value->isNegative()) return;

                builder.setInsertionPoint(elementRef);
                auto constOp = P4HIR::ConstOp::create(
                    builder, elementRef.getLoc(),
                    P4HIR::IntAttr::get(idxType, value->extOrTrunc(idxType.getWidth())));
                elementRef.getIndexMutable().assign(constOp.getResult());
            },
            numbering);
    }
}

static mlir::SymbolRefAttr createOOBRejectState(P4HIR::ParserOp parser) {
    auto *context = parser.getContext();
    mlir::OpBuilder builder(context);
    builder.setInsertionPoint(parser.getBody().front().getTerminator());
    auto state = P4HIR::ParserStateOp::create(
        builder, parser.getLoc(), "stateOutOfBound", mlir::DictionaryAttr());
    state.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&state.getBody().front());
    auto errorType = P4HIR::ErrorType::get(
        context, mlir::ArrayAttr::get(context, {mlir::StringAttr::get(context, "StackOutOfBounds")}));
    P4HIR::ParserRejectOp::create(builder, parser.getLoc(),
        P4HIR::ErrorCodeAttr::get(errorType, mlir::StringAttr::get(context, "StackOutOfBounds")));
    return state.getSymbolRef();
}

// Clone, substitute indices, and rewrite transitions into the unrolled parser.
static LogicalResult materializeUnrolled(P4HIR::ParserOp parser, SymbolicResult &symbolicResult,
                                         const SCCInfo &scc, StackNumbering &numbering) {
    if (symbolicResult.states.empty()) return success();

    bool hasOOB = false;
    for (auto &entry : symbolicResult.visitedMap)
        if (!entry.second) {
            hasOOB = true;
            break;
        }

    mlir::SymbolRefAttr rejectRef;
    if (hasOOB)
        rejectRef = createOOBRejectState(parser);

    if (failed(createClones(parser, symbolicResult, scc))) return failure();
    substituteConstantIndices(parser, symbolicResult, numbering);
    substituteExplicitIndices(parser, symbolicResult, numbering);
    return rewriteTransitions(parser, symbolicResult, scc, rejectRef, numbering);
}

struct ParserUnroll : public impl::ParserUnrollBase<ParserUnroll> {
    void runOnOperation() override {
        getOperation()->walk([&](P4HIR::ParserOp parser) {
            LLVM_DEBUG(llvm::dbgs() << "\n=== Parser Unroll: " << parser.getName() << " ===\n");

            auto backEdges = findBackEdges(parser);
            LLVM_DEBUG(llvm::dbgs() << "  back edges found: " << backEdges.size() << "\n");

            // Collect per-state {HSp} and declaration positions.
            StackNumbering numbering;
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

            auto scc =
                buildSCCInfo(parser, backEdges, stateAccesses, untrackable, declarationPos);
            LLVM_DEBUG(llvm::dbgs() << "  SCC members: " << scc.members.size() << " across "
                                    << scc.combinedByHead.size() << " loop(s)\n");

            auto symbolicResult =
                runSymbolicExecution(parser, scc, std::move(stateAccesses), numbering);
            if (failed(materializeUnrolled(parser, symbolicResult, scc, numbering)))
                signalPassFailure();
        });
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createParserUnrollPass() { return std::make_unique<ParserUnroll>(); }

}  // namespace P4::P4MLIR
