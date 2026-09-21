// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Transforms/ParserSymbolicExecution.h"

#include <deque>
#include <functional>

#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"

#define DEBUG_TYPE "p4hir-parser-unroll"

using namespace mlir;

namespace P4::P4MLIR {

// Build a stack id.
StackId StackId::make(mlir::MLIRContext *context, int64_t base,
                      llvm::ArrayRef<int64_t> path) {
    llvm::SmallVector<int64_t, 5> elements;
    elements.push_back(base);
    elements.append(path.begin(), path.end());
    return StackId(mlir::DenseI64ArrayAttr::get(context, elements));
}

bool StackId::operator<(const StackId &rhs) const {
    for (size_t i = 0, n = std::min(attribute.size(), rhs.attribute.size()); i < n;
         ++i) {
        if (attribute[i] != rhs.attribute[i]) return attribute[i] < rhs.attribute[i];
    }
    return attribute.size() < rhs.attribute.size();
}

// Print stack id value for debugging.
std::string StackId::render() const {
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    for (auto [index, field] : llvm::enumerate(attribute.asArrayRef())) {
        stream << (index == 0 ? "#" : ".") << field;
    }
    return rendered;
}

// Restrict a value map to the given keys.
ValueMap ValueMap::restrictTo(const llvm::DenseSet<StackId> &keep) const {
    ValueMap restricted;
    for (auto &entry : data)
        if (keep.contains(entry.first)) restricted.insert(entry);
    return restricted;
}

// Encode value map as sorted [stackId, value] ArrayAttr.
mlir::ArrayAttr ValueMap::encode(mlir::MLIRContext *context) const {
    auto entries = llvm::to_vector(data);
    llvm::sort(entries, llvm::less_first());
    llvm::SmallVector<mlir::Attribute> encoded;
    for (auto &entry : entries)
        encoded.push_back(mlir::ArrayAttr::get(context, {entry.first.asAttribute(), entry.second}));
    return mlir::ArrayAttr::get(context, encoded);
}

bool IndexMap::isOOBFor(const StackAccess &access) const {
    auto it = data.find(access.key);
    return it != data.end() && it->second + access.count > access.size;
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
    for (auto &access : accesses) result.data[access.key] += access.count;
    return result;
}

// Filter out IndexMap used for particular stacks. Used to filter out
// relevant data for particular SCC.
IndexMap IndexMap::restrictTo(llvm::ArrayRef<StackAccess> relevant) const {
    IndexMap result;
    for (auto &access : relevant) {
        auto it = data.find(access.key);
        if (it != data.end()) result.data[access.key] = it->second;
    }
    return result;
}

unsigned IndexMap::indexOf(StackId key) const {
    auto it = data.find(key);
    return it == data.end() ? 0 : it->second;
}

// Serializes and sorts the data_.
mlir::ArrayAttr IndexMap::encode(mlir::MLIRContext *context) const {
    auto entries = llvm::to_vector(data);
    llvm::sort(entries, llvm::less_first());
    llvm::SmallVector<mlir::Attribute> encoded;
    for (auto &entry : entries)
        encoded.push_back(mlir::ArrayAttr::get(
            context,
            {entry.first.asAttribute(),
             mlir::IntegerAttr::get(mlir::IndexType::get(context), entry.second)}));
    return mlir::ArrayAttr::get(context, encoded);
}

VisitedKey SymbolicResult::makeVisitedKey(mlir::MLIRContext *context,
                                          P4HIR::ParserStateOp state,
                                          const IndexMap &indexMap,
                                          const ValueMap &valueMap) {
    return {state,
            mlir::ArrayAttr::get(context,
                                 {indexMap.encode(context), valueMap.encode(context)})};
}

mlir::FailureOr<std::optional<unsigned>> SymbolicResult::lookupSuccessor(
    P4HIR::ParserStateOp state, const IndexMap &indexMap, const ValueMap &valueMap) const {
    auto it = visitedMap.find(makeVisitedKey(state.getContext(), state, indexMap, valueMap));
    if (it == visitedMap.end()) return mlir::failure();
    return it->second;
}

void StackNumbering::copyNumbering(mlir::Value original, mlir::Value cloned) {
    if (auto it = valueIds.find(original); it != valueIds.end())
        valueIds[cloned] = it->second;
}

// Stack variable id that a value refers to, if any.
mlir::FailureOr<StackId> StackNumbering::getStackId(mlir::Value value) {
    auto *context = value.getContext();
    llvm::SmallVector<int64_t, 4> reversePath;
    int64_t base = 0;

    while (true) {
        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            if (mlir::isa<P4HIR::ParserOp>(arg.getOwner()->getParentOp())) {
                base = forValue(arg);
                break;
            }
            // Fix cases where block arguments vary through branches.
            auto *block = arg.getOwner();
            unsigned argIndex = arg.getArgNumber();
            mlir::Value resolved;
            for (auto *pred : block->getPredecessors()) {
                auto branchOp =
                    mlir::dyn_cast<mlir::BranchOpInterface>(pred->getTerminator());
                if (!branchOp) return mlir::failure();
                unsigned succIndex = 0;
                for (unsigned i = 0, e = pred->getTerminator()->getNumSuccessors();
                     i < e; ++i) {
                    if (pred->getTerminator()->getSuccessor(i) == block) {
                        succIndex = i;
                        break;
                    }
                }
                mlir::Value forwarded =
                    branchOp.getSuccessorOperands(succIndex)[argIndex];
                if (!forwarded || (resolved && resolved != forwarded))
                    return mlir::failure();
                resolved = forwarded;
            }
            if (!resolved || resolved == arg) return mlir::failure();
            value = resolved;
            continue;
        }
        auto *definingOp = value.getDefiningOp();
        if (!definingOp) return mlir::failure();

        if (auto var = mlir::dyn_cast<P4HIR::VariableOp>(definingOp)) {
            base = forValue(var.getResult());
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
        // Handle value propogation through IfOps
        if (auto regionBranch =
                mlir::dyn_cast<mlir::RegionBranchOpInterface>(definingOp)) {
            auto result = mlir::cast<mlir::OpResult>(value);
            unsigned resultIndex = result.getResultNumber();
            llvm::SmallVector<mlir::Value> predecessorValues;
            regionBranch.getPredecessorValues(
                mlir::RegionSuccessor(definingOp, definingOp->getResults()),
                resultIndex, predecessorValues);
            if (predecessorValues.empty()) return mlir::failure();
            mlir::Value resolved = predecessorValues.front();
            if (!resolved || resolved == value) return mlir::failure();
            for (auto predecessorValue : llvm::drop_begin(predecessorValues))
                if (predecessorValue != resolved) return mlir::failure();
            value = resolved;
            continue;
        }
        return mlir::failure();
    }

    auto path = llvm::to_vector(llvm::reverse(reversePath));
    return StackId::make(context, base, path);
}

// Fold a value to constant attribute.
mlir::TypedAttr StackNumbering::foldToConstAttr(mlir::Value value, const ValueMap &valueMap) {
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) return {};

    if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(definingOp)) {
        if (auto key = getStackId(readOp.getRef()); succeeded(key)) {
            auto it = valueMap.find(*key);
            if (it != valueMap.end()) return it->second;
        }
        return {};
    }

    llvm::SmallVector<mlir::Attribute> operandConsts;
    for (mlir::Value operand : definingOp->getOperands()) {
        auto folded = foldToConstAttr(operand, valueMap);
        if (!folded) return {};
        operandConsts.push_back(folded);
    }
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::failed(definingOp->fold(operandConsts, results)) || results.size() != 1) return {};
    if (auto attr = results[0].dyn_cast<mlir::Attribute>())
        return mlir::dyn_cast<mlir::TypedAttr>(attr);
    if (auto foldedValue = results[0].dyn_cast<mlir::Value>())
        return foldToConstAttr(foldedValue, valueMap);
    return {};
}

// Fold a value to constant integer.
mlir::FailureOr<llvm::APSInt> StackNumbering::foldToConstInt(mlir::Value value,
                                                              const ValueMap &valueMap) {
    if (auto attribute = foldToConstAttr(value, valueMap))
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
            auto key = numbering.getStackId(assignOp.getRef());
            if (failed(key)) continue;
            if (auto attribute = numbering.foldToConstAttr(assignOp.getValue(), valueMap))
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
ValueMap StackNumbering::interpretState(
    P4HIR::ParserStateOp state, ValueMap valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess) {
    interpretBlock(*state.getBlock(), valueMap, onAccess, *this);
    return valueMap;
}

// Collect index-variable ids referenced by a value.
void StackNumbering::collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out) {
    if (auto readOp = value.getDefiningOp<P4HIR::ReadOp>()) {
        if (auto key = getStackId(readOp.getRef()); succeeded(key)) out.insert(*key);
        return;
    }
    if (auto *definingOp = value.getDefiningOp())
        for (mlir::Value operand : definingOp->getOperands())
            collectVarsInIndex(operand, out);
}

// Collect all index variables used across the parser.
llvm::DenseSet<StackId> StackNumbering::collectIndexVars(P4HIR::ParserOp parser) {
    llvm::DenseSet<StackId> out;
    std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &block) {
        for (auto &op : block) {
            if (auto arrayElementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op))
                collectVarsInIndex(arrayElementRef.getIndex(), out);
            else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op))
                for (auto &scopeBlock : scopeOp.getRegion())
                    visitBlock(scopeBlock);
        }
    };
    for (auto stateOp : parser.states())
        visitBlock(*stateOp.getBlock());
    return out;
}

// Assign same numbering to variables with same nameAttr in the different states.
void StackNumbering::canonicalizeParserVariables(P4HIR::ParserOp parser) {
    llvm::DenseMap<mlir::StringAttr, unsigned> nameIds;
    for (auto stateOp : parser.states()) {
        for (auto var : stateOp.getBlock()->getOps<P4HIR::VariableOp>()) {
            if (auto name = var.getName()) {
                auto nameAttr = mlir::StringAttr::get(var.getContext(), *name);
                auto [it, inserted] = nameIds.try_emplace(nameAttr, nextId);
                if (inserted) nextId++;
                valueIds[var.getResult()] = it->second;
            }
        }
    }
}

// Finds header stacks accessed via data[nextIndex] in a state. Returns failure
// if any stack identity can't be resolved to prevent incorrect unrolling.
mlir::FailureOr<llvm::SmallVector<StackAccess>> StackNumbering::computeStackAccesses(
    P4HIR::ParserStateOp state) {
    llvm::SmallVector<StackAccess> result;
    llvm::DenseSet<StackId> seen;
    llvm::DenseMap<StackId, unsigned> counts;
    bool unidentified = false;

    auto record = [&](mlir::Value input) {
        auto type = input.getType();
        if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type)) type = ref.getObjectType();
        auto stackType = mlir::dyn_cast<P4HIR::HeaderStackType>(type);
        if (!stackType || stackType.getArraySize() == 0) return;
        size_t size = stackType.getArraySize();
        auto key = getStackId(input);
        if (failed(key)) {
            unidentified = true;
            return;
        }
        if (!seen.insert(*key).second) return;
        result.push_back({*key, size});
    };

    auto visitElementRef = [&](P4HIR::ArrayElementRefOp elementRef) {
        mlir::Value idx = elementRef.getIndex();
        if (matchPattern(idx, m_Constant())) return;

        auto dataRef = elementRef.getInput().getDefiningOp<P4HIR::StructFieldRefOp>();
        if (!dataRef || dataRef.getFieldName() != P4HIR::HeaderStackType::dataFieldName) return;

        bool isNext = false;
        if (auto readOp = idx.getDefiningOp<P4HIR::ReadOp>()) {
            if (auto nextIndexRef = readOp.getRef().getDefiningOp<P4HIR::StructFieldRefOp>())
                isNext = nextIndexRef.getFieldName() == P4HIR::HeaderStackType::nextIndexFieldName;
        } else if (auto structExtract = idx.getDefiningOp<P4HIR::StructExtractOp>()) {
            isNext = structExtract.getFieldName() == P4HIR::HeaderStackType::nextIndexFieldName;
        }
        if (isNext)
            if (auto key = getStackId(dataRef.getInput()); succeeded(key))
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
        access.count = it == counts.end() ? 1 : it->second;
    }
    return result;
}

static bool setContains(mlir::Attribute setAttr, mlir::Attribute valueAttr) {
    if (mlir::isa<P4HIR::UniversalSetAttr>(setAttr)) return true;
    auto set = mlir::dyn_cast<P4HIR::SetAttr>(setAttr);
    if (!set || set.getKind() != P4HIR::SetKind::Constant) return false;
    assert(set.getMembers().size() == 1 && "expected single-member constant set");
    auto valueInt = P4HIR::getConstantInt(valueAttr);
    auto memberInt = P4HIR::getConstantInt(set.getMembers()[0]);
    return valueInt && memberInt && *memberInt == *valueInt;
}

SymbolicExecution::SymbolicExecution(
    P4HIR::ParserOp parser, AccessMap stateAccesses,
    StackNumbering &numbering,
    RelevantStacksProvider getRelevantStacks,
    unsigned limit,
    const llvm::DenseSet<P4HIR::ParserStateOp> &skipStates,
    const llvm::DenseSet<StackId> &extraIndexVars)
    : parser(parser), stateAccesses(std::move(stateAccesses)),
      numbering(numbering), getRelevantStacks(getRelevantStacks),
      limit(limit), skipStates(skipStates), extraIndexVars(extraIndexVars) {}

// Tries to fold transitionSelects, returns all successors if it doesn't succeed.
llvm::SmallVector<P4HIR::ParserStateOp> SymbolicExecution::resolveSuccessors(
    P4HIR::ParserStateOp state, const ValueMap &valueMap) {
    auto *terminator = state.getNextTransition();
    auto selectOp = mlir::dyn_cast<P4HIR::ParserTransitionSelectOp>(terminator);
    if (!selectOp) return llvm::to_vector(state.getNextStates());

    llvm::SmallVector<mlir::TypedAttr> foldedArgs;
    for (mlir::Value arg : selectOp.getArgs()) {
        auto folded = numbering.foldToConstAttr(arg, valueMap);
        if (!folded) return llvm::to_vector(state.getNextStates());
        foldedArgs.push_back(folded);
    }

    for (auto selectCase : selectOp.selects()) {
        auto selectKeys = selectCase.getSelectKeys();
        if (selectKeys.size() != foldedArgs.size())
            return llvm::to_vector(state.getNextStates());

        bool matches = true;
        for (auto [key, arg] : llvm::zip_equal(selectKeys, foldedArgs)) {
            auto foldedKey = numbering.foldToConstAttr(key, ValueMap{});
            if (!foldedKey || !setContains(foldedKey, arg)) {
                matches = false;
                break;
            }
        }
        if (matches) {
            auto targetState =
                parser.lookupSymbol<P4HIR::ParserStateOp>(selectCase.getStateAttr());
            if (!targetState) break;
            return {targetState};
        }
    }
    return llvm::to_vector(state.getNextStates());
}

// Resolve value map against nextIndex entries to contain interference
// between header stack derived indices and plain indices
ValueMap SymbolicExecution::buildResolveValueMap(P4HIR::ParserStateOp state,
                                                  const ValueMap &valueMap,
                                                  const IndexMap &indexMap) {
    ValueMap resolveMap = valueMap;
    for (auto &op : *state.getBlock()) {
        auto fieldRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(&op);
        if (!fieldRef || fieldRef.getFieldName() != P4HIR::HeaderStackType::nextIndexFieldName)
            continue;
        auto nextIdxStackId = numbering.getStackId(fieldRef.getResult());
        auto stackStackId = numbering.getStackId(fieldRef.getInput());
        if (failed(nextIdxStackId) || failed(stackStackId)) continue;
        unsigned index = indexMap.indexOf(*stackStackId);
        auto refType = mlir::cast<P4HIR::ReferenceType>(fieldRef.getType());
        auto indexType = refType.getObjectType();
        resolveMap[*nextIdxStackId] =
            P4HIR::IntAttr::get(indexType, static_cast<int64_t>(index));
    }
    return resolveMap;
}

// Traverse through states calculating correct indices for each loop iteration.
SymbolicResult SymbolicExecution::run() {
    SymbolicResult result;
    result.accesses = std::move(stateAccesses);
    result.indexVars = numbering.collectIndexVars(parser);
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
        if (++bfsIterations > limit) {
            parser.emitWarning()
                << "symbolic execution exceeded " << limit
                << " state instances; skipping unroll for parser '"
                << parser.getSymName() << "'";
            result.states.clear();
            result.visitedMap.clear();
            return result;
        }

        auto [state, indexMap, valueMap] = std::move(worklist.front());
        worklist.pop_front();

        if (state.isTerminal() || skipStates.contains(state)) continue;

        llvm::ArrayRef<StackAccess> relevant = getRelevantStacks(state);
        IndexMap restrictedIndexMap = indexMap.restrictTo(relevant);
        ValueMap restrictedValueMap = valueMap.restrictTo(result.indexVars);
        VisitedKey key = SymbolicResult::makeVisitedKey(parser.getContext(), state,
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
        ValueMap valueMapAfter = numbering.interpretState(
            state, valueMap, [](mlir::Operation *, const ValueMap &) {});

        ValueMap resolveMap = buildResolveValueMap(state, valueMap, indexMap);
        auto resolved = resolveSuccessors(state, resolveMap);
        llvm::DenseSet<P4HIR::ParserStateOp> resolvedSet(resolved.begin(), resolved.end());

        for (auto successor : state.getNextStates()) {
            if (successor.isTerminal()) continue;
            if (resolvedSet.contains(successor)) {
                worklist.push_back({successor, indexMapAfter, valueMapAfter});
                continue;
            }
            IndexMap prunedIndexMap =
                indexMapAfter.restrictTo(getRelevantStacks(successor));
            ValueMap prunedValueMap = valueMapAfter.restrictTo(result.indexVars);
            VisitedKey prunedKey =
                SymbolicResult::makeVisitedKey(parser.getContext(), successor,
                                               prunedIndexMap, prunedValueMap);
            result.visitedMap.try_emplace(prunedKey, std::nullopt);
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "  symbolic execution: " << result.states.size()
                            << " state instances, " << result.visitedMap.size()
                            << " visited keys\n");
    return result;
}

}  // namespace P4::P4MLIR
