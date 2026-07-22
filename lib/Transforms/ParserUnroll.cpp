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

// Core types - Definitions 2-5 of the algorithm.
// Definition 1 (the parser state graph) is P4HIR::ParserOp itself.

// Header-stack storage identity, encoded as an ArrayAttr [base, path...] so it
// keys llvm::DenseMap via MLIR's DenseMapInfo<Attribute>. The base is a
// value-numbered VariableOp/BlockArgument; the path is the struct field chain.
using StackId = mlir::Attribute;

// Build a stack id.
static StackId makeStackId(mlir::MLIRContext *context, unsigned base,
                           llvm::ArrayRef<uint32_t> path) {
    llvm::SmallVector<mlir::Attribute, 5> elements;
    auto indexType = mlir::IntegerType::get(context, 32);
    elements.push_back(mlir::IntegerAttr::get(indexType, base));
    for (uint32_t fieldIndex : path)
        elements.push_back(mlir::IntegerAttr::get(indexType, fieldIndex));
    return mlir::ArrayAttr::get(context, elements);
}

// Order two stack ids.
static bool stackIdLess(StackId lhs, StackId rhs) {
    auto lhsArray = mlir::cast<mlir::ArrayAttr>(lhs);
    auto rhsArray = mlir::cast<mlir::ArrayAttr>(rhs);
    for (size_t i = 0, n = std::min(lhsArray.size(), rhsArray.size()); i < n; ++i) {
        int64_t lhsField = mlir::cast<mlir::IntegerAttr>(lhsArray[i]).getInt();
        int64_t rhsField = mlir::cast<mlir::IntegerAttr>(rhsArray[i]).getInt();
        if (lhsField != rhsField) return lhsField < rhsField;
    }
    return lhsArray.size() < rhsArray.size();
}

// Value-numbers each stack base deterministically, on first encounter. Named
// variables are keyed by name so the base stays stable across state cloning
// (clones keep the name); block arguments and anonymous variables are keyed by
// their SSA value (block args are never cloned).
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
    for (auto [index, element] : llvm::enumerate(mlir::cast<mlir::ArrayAttr>(id))) {
        int64_t field = mlir::cast<mlir::IntegerAttr>(element).getInt();
        rendered += (index == 0 ? "#" : ".") + std::to_string(field);
    }
    return rendered;
}

// Symbolic value map V (Definition 2).
using ValueMap = llvm::DenseMap<StackId, mlir::TypedAttr>;

// Definition 5 / {HSp}: a header stack variable used in a state, plus its size.
struct StackAccess {
    StackId key;         // the stack variable (in {HSp})
    size_t size;         // OOB when the highest index used reaches size
    unsigned count = 1;  // number of .next accesses to this stack in the state
};

// Definition 2: symbolic index map M = {stack_key -> nextIndex}.
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
        llvm::SmallVector<mlir::Attribute> encoded;
        auto indexType = mlir::IntegerType::get(context, 32);
        for (auto &entry : entries)
            encoded.push_back(mlir::ArrayAttr::get(
                context, {entry.first, mlir::IntegerAttr::get(indexType, entry.second)}));
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

// Definition 3 / Definition 4: dedup key as an Attribute.
static mlir::Attribute makeVisitedKey(mlir::MLIRContext *context, mlir::StringAttr name,
                                      const IndexMap &indexMap, const ValueMap &valueMap) {
    return mlir::ArrayAttr::get(
        context, {name, indexMap.encode(context), encodeValueMap(context, valueMap)});
}

// Header-stack element count for a (reference) type, if it is a stack.
static std::optional<size_t> stackSizeOf(mlir::Type type) {
    if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type)) type = ref.getObjectType();
    if (auto stackType = mlir::dyn_cast<P4HIR::HeaderStackType>(type))
        return stackType.getArraySize();
    return std::nullopt;
}

// Stack variable id that a value refers to, if any.
static std::optional<StackId> getStackId(mlir::Value value, StackNumbering &numbering) {
    auto *context = value.getContext();
    llvm::SmallVector<uint32_t, 4> reversePath;
    unsigned base = 0;

    while (true) {
        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            base = numbering.forValue(arg);
            break;
        }
        auto *definingOp = value.getDefiningOp();
        if (!definingOp) return std::nullopt;

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
        return std::nullopt;
    }

    llvm::SmallVector<uint32_t, 4> path(reversePath.rbegin(), reversePath.rend());
    return makeStackId(context, base, path);
}

// Fold a value to constant attribute.
static mlir::TypedAttr evalConstAttr(mlir::Value value, const ValueMap &valueMap,
                                     StackNumbering &numbering) {
    mlir::Attribute attribute;
    if (mlir::matchPattern(value, mlir::m_Constant(&attribute)))
        return mlir::dyn_cast<mlir::TypedAttr>(attribute);

    auto *definingOp = value.getDefiningOp();
    if (!definingOp) return {};

    if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(definingOp)) {
        if (auto key = getStackId(readOp.getRef(), numbering)) {
            auto it = valueMap.find(*key);
            if (it != valueMap.end()) return it->second;
        }
        return {};
    }

    if (auto castOp = mlir::dyn_cast<P4HIR::CastOp>(definingOp)) {
        auto source = evalConstAttr(castOp.getSrc(), valueMap, numbering);
        if (!source) return {};
        mlir::Type destType = castOp.getType();
        if (mlir::isa<P4HIR::BitsType>(destType)) return P4HIR::foldConstantCast(destType, source);
        if (mlir::isa<P4HIR::InfIntType>(destType))
            if (auto sourceInt = P4HIR::getConstantInt(source)) {
                llvm::APInt widened = sourceInt->isUnsigned()
                                          ? sourceInt->zext(sourceInt->getBitWidth() + 1)
                                          : llvm::APInt(*sourceInt);
                return P4HIR::IntAttr::get(destType, widened);
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
    return mlir::dyn_cast_if_present<mlir::TypedAttr>(
        llvm::dyn_cast_if_present<mlir::Attribute>(results[0]));
}

// Fold a value to constant integer.
static std::optional<llvm::APSInt> evalConst(mlir::Value value, const ValueMap &valueMap,
                                             StackNumbering &numbering) {
    if (auto attribute = evalConstAttr(value, valueMap, numbering))
        return P4HIR::getConstantInt(attribute);
    return std::nullopt;
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
            if (!key) return;
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
        if (auto key = getStackId(readOp.getRef(), numbering)) out.insert(*key);
    } else if (auto castOp = value.getDefiningOp<P4HIR::CastOp>()) {
        collectVarsInIndex(castOp.getSrc(), out, numbering);
    } else if (auto binOp = value.getDefiningOp<P4HIR::BinOp>()) {
        collectVarsInIndex(binOp.getLhs(), out, numbering);
        collectVarsInIndex(binOp.getRhs(), out, numbering);
    }
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

// Whether a value flows into an array-element-ref index.
static bool flowsToArrayElementRefIndex(mlir::Value root) {
    llvm::SmallPtrSet<mlir::Value, 8> seen;
    llvm::SmallVector<mlir::Value, 8> worklist;
    worklist.push_back(root);
    while (!worklist.empty()) {
        mlir::Value current = worklist.pop_back_val();
        if (!seen.insert(current).second) continue;
        for (mlir::OpOperand &use : current.getUses()) {
            mlir::Operation *user = use.getOwner();
            if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(user)) {
                if (arrayElementRef.getIndex() == use.get()) return true;
                continue;
            }
            if (mlir::isa<P4HIR::ReadOp, P4HIR::CastOp>(user)) {
                if (user->getNumResults() == 1) worklist.push_back(user->getResult(0));
            }
        }
    }
    return false;
}

// Definition 5 / ParserStructure: builds {HSp} for a single state.
static std::optional<llvm::SmallVector<StackAccess>> computeStackAccesses(
    P4HIR::ParserStateOp state, StackNumbering &numbering) {
    llvm::SmallVector<StackAccess> result;
    llvm::DenseSet<StackId> seen;
    llvm::DenseMap<StackId, unsigned> counts;
    bool unidentified = false;

    auto record = [&](mlir::Value input) {
        auto size = stackSizeOf(input.getType());
        if (!size || *size == 0) return;
        auto key = getStackId(input, numbering);
        if (!key) {
            unidentified = true;
            return;
        }
        if (!seen.insert(*key).second) return;
        result.push_back({std::move(*key), *size});
    };

    state.walk([&](mlir::Operation *op) {
        if (auto fieldRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(op)) {
            if (fieldRef.getFieldName() == "nextIndex" &&
                flowsToArrayElementRefIndex(fieldRef.getResult()))
                record(fieldRef.getInput());
        } else if (auto structExtract = mlir::dyn_cast<P4HIR::StructExtractOp>(op)) {
            if (structExtract.getFieldName() == "nextIndex" &&
                flowsToArrayElementRefIndex(structExtract.getResult()))
                record(structExtract.getInput());
        } else if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(op)) {
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
                if (auto key = getStackId(dataRef.getInput(), numbering)) ++counts[*key];
            record(dataRef.getInput());
        }
    });

    if (unidentified) {
        mlir::emitWarning(state.getLoc(),
                          "cannot determine identity of header stack accessed in state '" +
                              state.getName().str() + "'; not unrolling this loop");
        return std::nullopt;
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
                             mlir::emitWarning(dest.getLoc(),
                                               "parser state '" + dest.getName().str() +
                                                   "' is unreachable from @start but is the head "
                                                   "of a cycle; parser-unroll will not process it");
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

// Declaration order of each parser state.
static llvm::DenseMap<P4HIR::ParserStateOp, unsigned> computedeclarationPositions(
    P4HIR::ParserOp parser) {
    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> declarationPos;
    unsigned position = 0;
    for (auto stateOp : parser.states()) declarationPos[stateOp] = position++;
    return declarationPos;
}

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
        mlir::emitWarning(loopHead.getLoc(),
                          "header stack '" + renderStackId(access.key) +
                              "' appears with conflicting sizes in the same SCC; "
                              "unroll depth may be incorrect");
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
        mlir::emitWarning(loopHead.getLoc(),
                          "parser loop at state '" + loopHead.getName().str() +
                              "' has no header stack operations; cannot infer unroll depth");
        return;
    }

    size_t minSize = std::numeric_limits<size_t>::max();
    for (auto &access : combined) minSize = std::min(minSize, access.size);
    if (minSize > kDefaultMaxUnrollDepth) {
        mlir::emitWarning(loopHead.getLoc(),
                          "parser loop at state '" + loopHead.getName().str() +
                              "' would unroll to depth " + std::to_string(minSize) + " (> " +
                              std::to_string(kDefaultMaxUnrollDepth) +
                              "); skipping. Reduce header stack size or raise the limit.");
        return;
    }

    for (auto stateOp : sccSet)
        if (scc.headOf.count(stateOp)) {
            mlir::emitWarning(loopHead.getLoc(),
                              "parser loop at state '" + loopHead.getName().str() +
                                  "' overlaps with a nested loop; outer loop will not be unrolled");
            return;
        }

    scc.combinedByHead[loopHead] = std::move(combined);
    for (auto stateOp : sccSet) {
        scc.members.insert(stateOp);
        scc.headOf[stateOp] = loopHead;
    }
}

// Loop heads reachable from a state.
static llvm::SmallVector<P4HIR::ParserStateOp> reachableHeads(P4HIR::ParserStateOp start,
                                                              const SCCInfo &scc) {
    llvm::SmallVector<P4HIR::ParserStateOp> heads;
    llvm::DenseSet<P4HIR::ParserStateOp> seenHeads;
    llvm::DenseSet<P4HIR::ParserStateOp> visited;
    llvm::SmallVector<P4HIR::ParserStateOp> worklist{start};
    while (!worklist.empty()) {
        auto current = worklist.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (current.isTerminal()) continue;
        if (auto it = scc.headOf.find(current); it != scc.headOf.end())
            if (seenHeads.insert(it->second).second) heads.push_back(it->second);
        for (auto next : current.getNextStates()) worklist.push_back(next);
    }
    return heads;
}

// Stacks reachable from a state (acyclic).
static llvm::SmallVector<StackAccess> reachableAccesses(P4HIR::ParserStateOp start,
                                                        const AccessMap &stateAccesses) {
    llvm::SmallVector<StackAccess> relevant;
    llvm::DenseSet<StackId> seenKeys;
    llvm::DenseSet<P4HIR::ParserStateOp> visited;
    llvm::SmallVector<P4HIR::ParserStateOp> worklist{start};
    while (!worklist.empty()) {
        auto current = worklist.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (current.isTerminal()) continue;
        if (auto it = stateAccesses.find(current); it != stateAccesses.end())
            for (auto &access : it->second)
                if (seenKeys.insert(access.key).second) relevant.push_back(access);
        for (auto next : current.getNextStates()) worklist.push_back(next);
    }
    return relevant;
}

// Per-state stacks that specialise clones.
static void computeRelevantStacks(P4HIR::ParserOp parser, SCCInfo &scc,
                                  const AccessMap &stateAccesses) {
    bool acyclic = scc.combinedByHead.empty();
    for (auto stateOp : parser.states()) {
        if (stateOp.isTerminal()) continue;
        llvm::SmallVector<StackAccess> relevant;
        if (acyclic) {
            relevant = reachableAccesses(stateOp, stateAccesses);
        } else if (auto it = scc.headOf.find(stateOp); it != scc.headOf.end()) {
            relevant = scc.combinedByHead[it->second];
        } else {
            llvm::DenseSet<StackId> seenKeys;
            for (auto head : reachableHeads(stateOp, scc)) {
                auto combinedIt = scc.combinedByHead.find(head);
                if (combinedIt == scc.combinedByHead.end()) continue;
                for (auto &access : combinedIt->second)
                    if (seenKeys.insert(access.key).second) relevant.push_back(access);
            }
        }
        scc.relevantStacks[stateOp] = std::move(relevant);
    }
}

// Build per-loop SCC info.
static SCCInfo buildSCCInfo(P4HIR::ParserOp parser, llvm::ArrayRef<BackEdge> backEdges,
                            const AccessMap &stateAccesses,
                            const llvm::DenseSet<P4HIR::ParserStateOp> &untrackable) {
    SCCInfo scc;
    auto declarationPos = computedeclarationPositions(parser);
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
            mlir::emitWarning(parser.getLoc(),
                              "parser-unroll: BFS exceeded " +
                                  std::to_string(kMaxBFSStateInstances) +
                                  " state instances; aborting unroll for parser '" +
                                  parser.getSymName().str() + "'");
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
                return getStackId(nextIndexRef.getInput(), numbering);
        } else if (auto structExtract = value.getDefiningOp<P4HIR::StructExtractOp>()) {
            if (structExtract.getFieldName() == "nextIndex")
                return getStackId(structExtract.getInput(), numbering);
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
                if (!value || value->isNegative()) return;

                builder.setInsertionPoint(elementRef);
                auto constOp = P4HIR::ConstOp::create(
                    builder, elementRef.getLoc(),
                    P4HIR::IntAttr::get(idxType, value->extOrTrunc(idxType.getWidth())));
                elementRef.getIndexMutable().assign(constOp.getResult());
            },
            numbering);
    }
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
    for (auto stateOp : parser.states())
        if (stateOp.isReject()) {
            rejectRef = stateOp.getSymbolRef();
            break;
        }
    if (hasOOB && !rejectRef)
        return parser.emitError("parser loop unrolling requires a @reject state");

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

            // Collect per-state {HSp}.
            StackNumbering numbering;
            AccessMap stateAccesses;
            llvm::DenseSet<P4HIR::ParserStateOp> untrackable;
            for (auto stateOp : parser.states()) {
                auto accesses = computeStackAccesses(stateOp, numbering);
                if (!accesses) {
                    untrackable.insert(stateOp);
                    stateAccesses[stateOp] = {};
                } else {
                    stateAccesses[stateOp] = std::move(*accesses);
                }
            }

            auto scc = buildSCCInfo(parser, backEdges, stateAccesses, untrackable);
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
