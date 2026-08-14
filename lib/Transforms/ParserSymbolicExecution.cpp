// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Transforms/ParserSymbolicExecution.h"

#include <deque>
#include <functional>

#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"

#define DEBUG_TYPE "p4hir-parser-unroll"

using namespace mlir;

namespace P4::P4MLIR {

static constexpr unsigned kMaxBFSStateInstances = 100000;

// Build a stack id.
StackId makeStackId(mlir::Builder &attrBuilder, int64_t base, llvm::ArrayRef<int64_t> path) {
    llvm::SmallVector<int64_t, 5> elements;
    elements.push_back(base);
    elements.append(path.begin(), path.end());
    return attrBuilder.getDenseI64ArrayAttr(elements);
}

// Order two stack ids.
bool stackIdLess(StackId lhs, StackId rhs) {
    auto lhsArr = mlir::cast<mlir::DenseI64ArrayAttr>(lhs);
    auto rhsArr = mlir::cast<mlir::DenseI64ArrayAttr>(rhs);
    for (size_t i = 0, n = std::min(lhsArr.size(), rhsArr.size()); i < n; ++i) {
        if (lhsArr[i] != rhsArr[i]) return lhsArr[i] < rhsArr[i];
    }
    return lhsArr.size() < rhsArr.size();
}

// Print stack id value for debugging.
std::string renderStackId(StackId id) {
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    for (auto [index, field] :
         llvm::enumerate(mlir::cast<mlir::DenseI64ArrayAttr>(id).asArrayRef())) {
        stream << (index == 0 ? "#" : ".") << field;
    }
    return rendered;
}

bool IndexMap::isOOBFor(const StackAccess &access) const {
    auto it = data_.find(access.key);
    return it != data_.end() && it->second + access.count > access.size;
}

bool IndexMap::isOOBForAny(llvm::ArrayRef<StackAccess> accesses) const {
    for (auto &access : accesses)
        if (isOOBFor(access)) return true;
    return false;
}

// Increase all the stack indices by corresponding count. Used to
// increment the map while moving through i.e. through loop clones.
IndexMap IndexMap::advanced(llvm::ArrayRef<StackAccess> accesses) const {
    IndexMap result = *this;
    for (auto &access : accesses) result.data_[access.key] += access.count;
    return result;
}

// Filter out IndexMap used for particular stacks. Used to filter out
// relevant data for particular SCC.
IndexMap IndexMap::restrictTo(llvm::ArrayRef<StackAccess> relevant) const {
    IndexMap result;
    for (auto &access : relevant) {
        auto it = data_.find(access.key);
        if (it != data_.end()) result.data_[access.key] = it->second;
    }
    return result;
}

unsigned IndexMap::indexOf(StackId key) const {
    auto it = data_.find(key);
    return it == data_.end() ? 0 : it->second;
}

// Serializes and sorts the data_.
mlir::ArrayAttr IndexMap::encode(mlir::MLIRContext *context) const {
    auto entries = llvm::to_vector(data_);
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

// Encode value map as sorted [stackId, value] ArrayAttr.
static mlir::ArrayAttr encodeValueMap(mlir::MLIRContext *context, const ValueMap &valueMap) {
    auto entries = llvm::to_vector(valueMap);
    llvm::sort(entries,
               [](const auto &lhs, const auto &rhs) { return stackIdLess(lhs.first, rhs.first); });
    llvm::SmallVector<mlir::Attribute> encoded;
    for (auto &entry : entries)
        encoded.push_back(mlir::ArrayAttr::get(context, {entry.first, entry.second}));
    return mlir::ArrayAttr::get(context, encoded);
}

VisitedKey makeVisitedKey(mlir::MLIRContext *context, P4HIR::ParserStateOp state,
                          const IndexMap &indexMap, const ValueMap &valueMap) {
    return {state,
            mlir::ArrayAttr::get(context,
                                 {indexMap.encode(context), encodeValueMap(context, valueMap)})};
}

SymbolicResult::SuccessorLookup SymbolicResult::lookupSuccessor(
    P4HIR::ParserStateOp state, const IndexMap &indexMap, const ValueMap &valueMap) const {
    auto it = visitedMap.find(makeVisitedKey(state.getContext(), state, indexMap, valueMap));
    if (it == visitedMap.end()) return {};
    return {true, it->second};
}

// Header-stack element count for a (reference) type, if it is a stack.
mlir::FailureOr<size_t> stackSizeOf(mlir::Type type) {
    if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type)) type = ref.getObjectType();
    if (auto stackType = mlir::dyn_cast<P4HIR::HeaderStackType>(type))
        return stackType.getArraySize();
    return mlir::failure();
}

// Stack variable id that a value refers to, if any.
mlir::FailureOr<StackId> getStackId(mlir::Value value, StackNumbering &numbering) {
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

    auto path = llvm::to_vector(llvm::reverse(reversePath));
    return makeStackId(attrBuilder, base, path);
}

// Fold a value to constant attribute.
mlir::TypedAttr foldToConstAttr(mlir::Value value, const ValueMap &valueMap,
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
        auto folded = foldToConstAttr(operand, valueMap, numbering);
        if (!folded) return {};
        operandConsts.push_back(folded);
    }
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::failed(definingOp->fold(operandConsts, results)) || results.size() != 1) return {};
    if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(results[0]))
        return mlir::dyn_cast_if_present<mlir::TypedAttr>(attr);
    if (auto foldedValue = llvm::dyn_cast_if_present<mlir::Value>(results[0]))
        return foldToConstAttr(foldedValue, valueMap, numbering);
    return {};
}

// Fold a value to constant integer.
mlir::FailureOr<llvm::APSInt> foldToConstInt(mlir::Value value, const ValueMap &valueMap,
                                             StackNumbering &numbering) {
    if (auto attribute = foldToConstAttr(value, valueMap, numbering))
        if (auto constInt = P4HIR::getConstantInt(attribute))
            return *constInt;
    return mlir::failure();
}

// Collects array accesses and tries to fold assigns.
static void interpretBlock(
    mlir::Block &block, ValueMap &valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
    StackNumbering &numbering) {
    for (auto &op : block) {
        if (mlir::isa<P4HIR::ArrayElementRefOp, P4HIR::ArrayGetOp>(&op)) {
            onAccess(&op, valueMap);
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(&op)) {
            auto key = getStackId(assignOp.getRef(), numbering);
            if (failed(key)) continue;
            if (auto attribute = foldToConstAttr(assignOp.getValue(), valueMap, numbering))
                valueMap[*key] = attribute;
            else
                valueMap.erase(*key);
        } else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op)) {
            for (auto &scopeBlock : scopeOp.getRegion())
                interpretBlock(scopeBlock, valueMap, onAccess, numbering);
        }
    }
}

// Symbolically interpret a state's body, updating the value map.
ValueMap interpretState(
    P4HIR::ParserStateOp state, ValueMap valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
    StackNumbering &numbering) {
    interpretBlock(*state.getBlock(), valueMap, onAccess, numbering);
    return valueMap;
}

// Collect index-variable ids referenced by a value.
void collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out,
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
llvm::DenseSet<StackId> collectIndexVars(P4HIR::ParserOp parser, StackNumbering &numbering) {
    llvm::DenseSet<StackId> out;
    std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &block) {
        for (auto &op : block) {
            if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op))
                collectVarsInIndex(arrayElementRef.getIndex(), out, numbering);
            else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op))
                for (auto &scopeBlock : scopeOp.getRegion())
                    visitBlock(scopeBlock);
        }
    };
    for (auto stateOp : parser.states())
        visitBlock(*stateOp.getBlock());
    return out;
}

// Restrict a value map to the given keys.
ValueMap restrictValueMap(const ValueMap &valueMap, const llvm::DenseSet<StackId> &keep) {
    ValueMap restricted;
    for (auto &entry : valueMap)
        if (keep.contains(entry.first)) restricted.insert(entry);
    return restricted;
}

// Variable canonicalization
void canonicalizeParserVariables(P4HIR::ParserOp parser, StackNumbering &numbering) {
    llvm::DenseMap<mlir::StringAttr, unsigned> nameIds;
    for (auto stateOp : parser.states()) {
        for (auto var : stateOp.getBlock()->getOps<P4HIR::VariableOp>()) {
            if (auto name = var.getName()) {
                auto nameAttr = mlir::StringAttr::get(var.getContext(), *name);
                auto [it, inserted] = nameIds.try_emplace(nameAttr, numbering.nextId);
                if (inserted) numbering.nextId++;
                numbering.valueIds[var.getResult()] = it->second;
            }
        }
    }
}

// Stack access computation
mlir::FailureOr<llvm::SmallVector<StackAccess>> computeStackAccesses(
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

    auto visitElementRef = [&](P4HIR::ArrayElementRefOp elementRef) {
        mlir::Value idx = elementRef.getIndex();
        if (matchPattern(idx, m_Constant())) return;

        auto *arrayDef = elementRef.getInput().getDefiningOp();
        if (!arrayDef) return;
        auto dataRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(arrayDef);
        if (!dataRef || dataRef.getFieldName() != "data") return;

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
    };

    std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &block) {
        for (auto &op : block) {
            if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op))
                visitElementRef(elementRef);
            else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op))
                for (auto &scopeBlock : scopeOp.getRegion())
                    visitBlock(scopeBlock);
        }
    };
    visitBlock(*state.getBlock());

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

static bool setContains(mlir::Attribute setAttr, mlir::Attribute valueAttr) {
    if (mlir::isa<P4HIR::UniversalSetAttr>(setAttr)) return true;
    auto set = mlir::dyn_cast<P4HIR::SetAttr>(setAttr);
    if (!set || set.getKind() != P4HIR::SetKind::Constant) return false;
    auto valueInt = P4HIR::getConstantInt(valueAttr);
    if (!valueInt) return false;
    for (auto member : set.getMembers())
        if (auto memberInt = P4HIR::getConstantInt(member); memberInt && *memberInt == *valueInt)
            return true;
    return false;
}

// Tries to fold transitionSelects, returns all successors if it doesn't succeed.
static llvm::SmallVector<P4HIR::ParserStateOp> resolveSuccessors(P4HIR::ParserStateOp state,
                                                                 P4HIR::ParserOp parser,
                                                                 const ValueMap &valueMap,
                                                                 StackNumbering &numbering) {
    auto *terminator = state.getNextTransition();
    auto selectOp = mlir::dyn_cast<P4HIR::ParserTransitionSelectOp>(terminator);
    if (!selectOp) return llvm::to_vector(state.getNextStates());

    llvm::SmallVector<mlir::TypedAttr> foldedArgs;
    for (mlir::Value arg : selectOp.getArgs()) {
        auto folded = foldToConstAttr(arg, valueMap, numbering);
        if (!folded) return llvm::to_vector(state.getNextStates());
        foldedArgs.push_back(folded);
    }

    for (auto selectCase : selectOp.selects()) {
        auto selectKeys = selectCase.getSelectKeys();
        if (selectKeys.size() != foldedArgs.size())
            return llvm::to_vector(state.getNextStates());

        bool matches = true;
        for (auto [key, arg] : llvm::zip(selectKeys, foldedArgs)) {
            auto foldedKey = foldToConstAttr(key, ValueMap{}, numbering);
            if (!foldedKey || !setContains(foldedKey, arg)) {
                matches = false;
                break;
            }
        }
        if (matches) {
            auto targetState = parser.lookupSymbol<P4HIR::ParserStateOp>(selectCase.getStateAttr());
            if (!targetState) break;
            return {targetState};
        }
    }
    return llvm::to_vector(state.getNextStates());
}

// Traverse through states calculating correct indices for each loop iteration.
SymbolicResult runSymbolicExecution(P4HIR::ParserOp parser, AccessMap stateAccesses,
                                    StackNumbering &numbering,
                                    RelevantStacksProvider getRelevantStacks,
                                    CounterOnlyCheck isCounterOnly,
                                    const llvm::DenseSet<StackId> &extraIndexVars) {
    SymbolicResult result;
    result.accesses = std::move(stateAccesses);
    result.indexVars = collectIndexVars(parser, numbering);
    result.indexVars.insert(extraIndexVars.begin(), extraIndexVars.end());

    auto startState = parser.getStartState();
    if (!startState) return result;

    struct WorkItem {
        P4HIR::ParserStateOp state;
        IndexMap indexMap;
        ValueMap valueMap;
    };

    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> callsCount;

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

        llvm::ArrayRef<StackAccess> relevant = getRelevantStacks(state);
        IndexMap restrictedIndexMap = indexMap.restrictTo(relevant);
        ValueMap restrictedValueMap = restrictValueMap(valueMap, result.indexVars);
        VisitedKey key = makeVisitedKey(parser.getContext(), state,
                                       restrictedIndexMap, restrictedValueMap);

        auto accessesIt = result.accesses.find(state);
        assert(accessesIt != result.accesses.end() && "state missing from accesses map");
        bool oob = indexMap.isOOBForAny(accessesIt->second);

        if (oob) {
            result.visitedMap.try_emplace(key, std::nullopt);
            continue;
        }

        unsigned &countRef = callsCount[state];
        auto [visitedIt, inserted] = result.visitedMap.try_emplace(key, countRef);
        if (!inserted) continue;
        unsigned idx = countRef++;

        result.states.push_back({state, idx, indexMap, valueMap, {}});
        LLVM_DEBUG(llvm::dbgs() << "  visit " << state.getSymName() << " idx=" << idx << "\n");

        IndexMap indexMapAfter = indexMap.advanced(accessesIt->second);
        ValueMap valueMapAfter = interpretState(
            state, valueMap, [](mlir::Operation *, const ValueMap &) {}, numbering);

        if (isCounterOnly(state)) {
            auto resolved = resolveSuccessors(state, parser, valueMapAfter, numbering);
            llvm::DenseSet<P4HIR::ParserStateOp> resolvedSet(resolved.begin(), resolved.end());

            for (auto successor : state.getNextStates()) {
                if (successor.isTerminal()) continue;
                if (resolvedSet.contains(successor)) {
                    worklist.push_back({successor, indexMapAfter, valueMapAfter});
                    continue;
                }
                IndexMap prunedIndexMap =
                    indexMapAfter.restrictTo(getRelevantStacks(successor));
                ValueMap prunedValueMap = restrictValueMap(valueMapAfter, result.indexVars);
                VisitedKey prunedKey =
                    makeVisitedKey(parser.getContext(), successor, prunedIndexMap,
                                   prunedValueMap);
                result.visitedMap.try_emplace(prunedKey, std::nullopt);
            }
        } else {
            for (auto successor : state.getNextStates()) {
                if (successor.isTerminal()) continue;
                worklist.push_back({successor, indexMapAfter, valueMapAfter});
            }
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "  symbolic execution: " << result.states.size()
                            << " state instances, " << result.visitedMap.size()
                            << " visited keys\n");
    return result;
}

}  // namespace P4::P4MLIR
