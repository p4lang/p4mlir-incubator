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

bool StackId::overlaps(const StackId &rhs) const {
    size_t commonSize = std::min(attribute.size(), rhs.attribute.size());
    return attribute.asArrayRef().take_front(commonSize) ==
           rhs.attribute.asArrayRef().take_front(commonSize);
}

// Stack id of a field nested in this one.
StackId StackId::child(int64_t field) const {
    auto elements = llvm::to_vector(attribute.asArrayRef());
    elements.push_back(field);
    return StackId(mlir::DenseI64ArrayAttr::get(attribute.getContext(), elements));
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

// Restrict a value map to the entries overlapping the given keys.
ValueMap ValueMap::restrictTo(const llvm::DenseSet<StackId> &keep) const {
    ValueMap restricted;
    for (auto &entry : data)
        if (llvm::any_of(keep, [&](StackId keepKey) { return keepKey.overlaps(entry.first); }))
            restricted.insert(entry);
    return restricted;
}

// Keep only the values both maps agree on.
ValueMap ValueMap::intersect(const ValueMap &other) const {
    ValueMap merged;
    for (auto &entry : data) {
        auto it = other.data.find(entry.first);
        if (it != other.data.end() && it->second == entry.second) merged.insert(entry);
    }
    merged.snapshots = snapshots;
    merged.snapshots.insert(other.snapshots.begin(), other.snapshots.end());
    return merged;
}

std::optional<mlir::TypedAttr> ValueMap::findSnapshot(mlir::Value value) const {
    auto it = snapshots.find(value);
    if (it == snapshots.end()) return std::nullopt;
    return it->second;
}

void ValueMap::eraseOverlapping(StackId key) {
    llvm::SmallVector<StackId> overlapping;
    for (auto &entry : data)
        if (entry.first.overlaps(key)) overlapping.push_back(entry.first);
    for (auto stackId : overlapping) data.erase(stackId);
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
    for (auto &access : accesses)
        if (access.count) result.data[access.key] += access.count;
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
    if (auto snapshot = valueMap.findSnapshot(value)) return *snapshot;

    auto *definingOp = value.getDefiningOp();
    if (!definingOp) return {};

    if (mlir::isa<P4HIR::ReadOp, P4HIR::StructExtractOp>(definingOp)) {
        if (auto key = getStackId(value); succeeded(key)) {
            // A value stored through a declaration of another type is unknown here.
            auto it = valueMap.find(*key);
            if (it != valueMap.end())
                return it->second.getType() == value.getType() ? it->second : mlir::TypedAttr();
        }
        if (mlir::isa<P4HIR::ReadOp>(definingOp)) return {};
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

// Increment a known nextIndex value after a .next access.
static void advanceNextIndex(ValueMap &valueMap, StackId nextIndexKey) {
    auto it = valueMap.find(nextIndexKey);
    if (it == valueMap.end()) return;
    auto current = P4HIR::getConstantInt(it->second);
    if (!current) {
        valueMap.erase(nextIndexKey);
        return;
    }
    llvm::APSInt advanced = *current;
    ++advanced;
    it->second = P4HIR::IntAttr::get(it->second.getType(), advanced);
}

static void interpretBlock(
    mlir::Block &block, ValueMap &valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
    StackNumbering &numbering);

static ValueMap interpretRegion(
    mlir::Region &region, ValueMap valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
    StackNumbering &numbering) {
    for (auto &regionBlock : region) interpretBlock(regionBlock, valueMap, onAccess, numbering);
    return valueMap;
}

// Collects array accesses, folds assigns and records the value of every read.
static void interpretBlock(
    mlir::Block &block, ValueMap &valueMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
    StackNumbering &numbering) {
    for (auto &op : block) {
        if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op)) {
            auto nextIndexKey = numbering.getNextIndexKey(elementRef);
            onAccess(&op, valueMap);
            if (succeeded(nextIndexKey)) advanceNextIndex(valueMap, *nextIndexKey);
        } else if (mlir::isa<P4HIR::ArrayGetOp>(&op)) {
            onAccess(&op, valueMap);
        } else if (mlir::isa<P4HIR::ReadOp, P4HIR::StructExtractOp>(&op)) {
            mlir::Value result = op.getResult(0);
            valueMap.recordSnapshot(result, numbering.foldToConstAttr(result, valueMap));
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(&op)) {
            auto key = numbering.getStackId(assignOp.getRef());
            if (failed(key)) continue;
            auto attribute = numbering.foldToConstAttr(assignOp.getValue(), valueMap);
            valueMap.eraseOverlapping(*key);
            if (attribute) valueMap[*key] = attribute;
        } else if (auto assignSliceOp = mlir::dyn_cast<P4HIR::AssignSliceOp>(&op)) {
            if (auto key = numbering.getStackId(assignSliceOp.getRef()); succeeded(key))
                valueMap.eraseOverlapping(*key);
        } else if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(&op)) {
            for (auto &scopeBlock : scopeOp.getRegion())
                interpretBlock(scopeBlock, valueMap, onAccess, numbering);
        } else if (auto ifOp = mlir::dyn_cast<P4HIR::IfOp>(&op)) {
            valueMap =
                interpretRegion(ifOp.getThenRegion(), valueMap, onAccess, numbering)
                    .intersect(interpretRegion(ifOp.getElseRegion(), valueMap, onAccess, numbering));
        } else if (op.getNumRegions() != 0) {
            // Regions may run any number of times, including not at all.
            ValueMap merged = valueMap;
            for (auto &region : op.getRegions())
                merged = merged.intersect(interpretRegion(region, valueMap, onAccess, numbering));
            valueMap = std::move(merged);
        } else if (!mlir::isa<P4HIR::StructFieldRefOp>(&op)) {
            // Any other op taking a reference, e.g. a call with an inout argument, may write it.
            for (mlir::Value operand : op.getOperands())
                if (mlir::isa<P4HIR::ReferenceType>(operand.getType()))
                    if (auto key = numbering.getStackId(operand); succeeded(key))
                        valueMap.eraseOverlapping(*key);
        }
    }
}

// Symbolically interpret a state's body, updating the value map. nextIndex of
// every known stack is visible during interpretation but is not kept in the result.
ValueMap StackNumbering::interpretState(
    P4HIR::ParserStateOp state, ValueMap valueMap, const IndexMap &indexMap,
    llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess) {
    valueMap.clearSnapshots();
    for (auto &fields : stacks)
        valueMap[fields.nextIndexKey] = P4HIR::IntAttr::get(
            fields.nextIndexType, static_cast<int64_t>(indexMap.indexOf(fields.stackKey)));
    interpretBlock(*state.getBlock(), valueMap, onAccess, *this);
    for (auto &fields : stacks) valueMap.erase(fields.nextIndexKey);
    return valueMap;
}

// Collect index-variable ids referenced by a value.
void StackNumbering::collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out) {
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) return;
    if (mlir::isa<P4HIR::ReadOp, P4HIR::StructExtractOp>(definingOp)) {
        if (auto key = getStackId(value); succeeded(key)) {
            out.insert(*key);
            return;
        }
        if (mlir::isa<P4HIR::ReadOp>(definingOp)) return;
    }
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

// Finds header stacks a state depends on: accessed with a non-constant index or
// whose nextIndex is read. Only data[nextIndex] accesses advance the stack.
// Returns failure if any stack identity can't be resolved to prevent incorrect unrolling.
mlir::FailureOr<llvm::SmallVector<StackAccess>> StackNumbering::computeStackAccesses(
    P4HIR::ParserStateOp state) {
    llvm::SmallVector<StackAccess> result;
    llvm::DenseSet<StackId> seen;
    llvm::DenseMap<StackId, unsigned> counts;
    bool unidentified = false;

    auto stackTypeOf = [](mlir::Value input) {
        auto type = input.getType();
        if (auto ref = mlir::dyn_cast<P4HIR::ReferenceType>(type)) type = ref.getObjectType();
        return mlir::dyn_cast<P4HIR::HeaderStackType>(type);
    };

    auto record = [&](mlir::Value input) -> std::optional<StackFields> {
        auto stackType = stackTypeOf(input);
        if (!stackType || stackType.getArraySize() == 0) return std::nullopt;
        auto key = getStackId(input);
        if (failed(key)) {
            unidentified = true;
            return std::nullopt;
        }
        if (seen.insert(*key).second) result.push_back({*key, stackType.getArraySize()});
        return registerStack(*key, stackType);
    };

    auto visitElementRef = [&](P4HIR::ArrayElementRefOp elementRef) {
        mlir::Value index = elementRef.getIndex();
        if (matchPattern(index, m_Constant())) return;

        auto dataRef = elementRef.getInput().getDefiningOp<P4HIR::StructFieldRefOp>();
        if (!dataRef) return;
        auto fields = record(dataRef.getInput());
        if (!fields) return;
        if (auto indexKey = getStackId(index);
            succeeded(indexKey) && *indexKey == fields->nextIndexKey)
            ++counts[fields->stackKey];
    };

    auto visitFieldAccess = [&](mlir::Value input, unsigned fieldIndex) {
        auto stackType = stackTypeOf(input);
        if (stackType && fieldIndex + 1 == stackType.getElements().size() &&
            succeeded(getStackId(input)))
            record(input);
    };

    std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &block) {
        for (auto &op : block) {
            if (auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(&op))
                visitElementRef(elementRef);
            else if (auto fieldRef = mlir::dyn_cast<P4HIR::StructFieldRefOp>(&op))
                visitFieldAccess(fieldRef.getInput(), fieldRef.getFieldIndex());
            else if (auto structExtract = mlir::dyn_cast<P4HIR::StructExtractOp>(&op))
                visitFieldAccess(structExtract.getInput(), structExtract.getFieldIndex());
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
    for (auto &access : result) access.count = counts.lookup(access.key);
    return result;
}

// Remember where data and nextIndex of a stack live. The header stack verifier
// fixes data as the first field and nextIndex as the last one.
StackFields StackNumbering::registerStack(StackId stackKey, P4HIR::HeaderStackType stackType) {
    auto existing = llvm::find_if(
        stacks, [&](const StackFields &fields) { return fields.stackKey == stackKey; });
    if (existing != stacks.end()) return *existing;
    auto elements = stackType.getElements();
    StackFields fields{stackKey, stackKey.child(0),
                       stackKey.child(static_cast<int64_t>(elements.size() - 1)),
                       elements.back().type};
    stacks.push_back(fields);
    return fields;
}

// nextIndex key of the stack if the access is data[nextIndex], i.e. a .next access.
mlir::FailureOr<StackId> StackNumbering::getNextIndexKey(P4HIR::ArrayElementRefOp elementRef) {
    auto dataKey = getStackId(elementRef.getInput());
    auto indexKey = getStackId(elementRef.getIndex());
    if (failed(dataKey) || failed(indexKey)) return mlir::failure();
    for (auto &fields : stacks)
        if (fields.dataKey == *dataKey && fields.nextIndexKey == *indexKey)
            return fields.nextIndexKey;
    return mlir::failure();
}

bool StackNumbering::isNextIndexValue(mlir::Value value) {
    auto key = getStackId(value);
    return succeeded(key) && llvm::any_of(stacks, [&](const StackFields &fields) {
               return fields.nextIndexKey == *key;
           });
}

// Header stack element written at an index that folds outside of the stack.
bool StackNumbering::isOutOfBounds(P4HIR::ArrayElementRefOp elementRef,
                                   const ValueMap &valueMap) {
    auto dataKey = getStackId(elementRef.getInput());
    if (failed(dataKey) || llvm::none_of(stacks, [&](const StackFields &fields) {
            return fields.dataKey == *dataKey;
        }))
        return false;
    auto index = foldToConstInt(elementRef.getIndex(), valueMap);
    if (failed(index)) return false;
    auto arrayType = mlir::cast<P4HIR::ArrayType>(
        mlir::cast<P4HIR::ReferenceType>(elementRef.getInput().getType()).getObjectType());
    return index->isNegative() || index->getLimitedValue() >= arrayType.getSize();
}

static std::optional<bool> setContains(mlir::Attribute setAttr, mlir::Attribute valueAttr) {
    if (mlir::isa<P4HIR::UniversalSetAttr>(setAttr)) return true;
    auto set = mlir::dyn_cast<P4HIR::SetAttr>(setAttr);
    auto valueInt = P4HIR::getConstantInt(valueAttr);
    if (!set || !valueInt) return std::nullopt;
    switch (set.getKind()) {
        case P4HIR::SetKind::Constant: {
            assert(set.getMembers().size() == 1 && "expected single-member constant set");
            auto memberInt = P4HIR::getConstantInt(set.getMembers()[0]);
            if (!memberInt) return std::nullopt;
            return *memberInt == *valueInt;
        }
        case P4HIR::SetKind::Range: {
            auto lowInt = P4HIR::getConstantInt(set.getMembers()[0]);
            auto highInt = P4HIR::getConstantInt(set.getMembers()[1]);
            if (!lowInt || !highInt) return std::nullopt;
            return llvm::APSInt::compareValues(*lowInt, *valueInt) <= 0 &&
                   llvm::APSInt::compareValues(*valueInt, *highInt) <= 0;
        }
        case P4HIR::SetKind::Mask: {
            auto keyInt = P4HIR::getConstantInt(set.getMembers()[0]);
            auto maskInt = P4HIR::getConstantInt(set.getMembers()[1]);
            if (!keyInt || !maskInt || keyInt->getBitWidth() != valueInt->getBitWidth() ||
                maskInt->getBitWidth() != valueInt->getBitWidth())
                return std::nullopt;
            llvm::APInt value = *valueInt, key = *keyInt, mask = *maskInt;
            return ((value ^ key) & mask).isZero();
        }
        default:
            return std::nullopt;
    }
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
            if (!foldedKey) return llvm::to_vector(state.getNextStates());
            auto contained = setContains(foldedKey, arg);
            if (!contained) return llvm::to_vector(state.getNextStates());
            if (!*contained) {
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
        bool generic = false;
    };

    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> callsCount;

    std::deque<WorkItem> worklist;
    worklist.push_back({startState, IndexMap(), ValueMap()});

    unsigned instanceCount = 0;

    while (!worklist.empty()) {
        auto [state, indexMap, valueMap, generic] = std::move(worklist.front());
        worklist.pop_front();

        if (state.isTerminal()) continue;

        // Skipped loops leave stack positions unknown in every state they reach.
        if (generic || skipStates.contains(state)) {
            if (!result.genericStates.insert(state).second) continue;
            for (auto successor : state.getNextStates())
                worklist.push_back({successor, IndexMap(), ValueMap(), true});
            continue;
        }

        llvm::ArrayRef<StackAccess> relevant = getRelevantStacks(state);
        IndexMap restrictedIndexMap = indexMap.restrictTo(relevant);
        ValueMap restrictedValueMap = valueMap.restrictTo(result.indexVars);
        VisitedKey key = SymbolicResult::makeVisitedKey(parser.getContext(), state,
                                                        restrictedIndexMap, restrictedValueMap);
        if (result.visitedMap.contains(key)) continue;

        auto accessesIt = result.accesses.find(state);
        assert(accessesIt != result.accesses.end() && "state missing from accesses map");
        bool oob = indexMap.isOOBForAny(accessesIt->second);

        ValueMap valueMapAfter;
        if (!oob)
            valueMapAfter = numbering.interpretState(
                state, valueMap, indexMap,
                [&](mlir::Operation *accessOp, const ValueMap &accessValueMap) {
                    auto elementRef = mlir::dyn_cast<P4HIR::ArrayElementRefOp>(accessOp);
                    if (elementRef && numbering.isOutOfBounds(elementRef, accessValueMap))
                        oob = true;
                });

        if (oob) {
            result.visitedMap.try_emplace(key, std::nullopt);
            result.outOfBoundIndex.try_emplace(key, result.outOfBoundStates.size());
            result.outOfBoundStates.push_back({state, 0, indexMap, valueMap, {}, {}});
            continue;
        }

        if (++instanceCount > limit) {
            parser.emitWarning()
                << "symbolic execution exceeded " << limit
                << " state instances; skipping unroll for parser '"
                << parser.getSymName() << "'";
            result.states.clear();
            result.visitedMap.clear();
            result.outOfBoundStates.clear();
            result.outOfBoundIndex.clear();
            return result;
        }

        unsigned &countRef = callsCount[state];
        result.visitedMap.try_emplace(key, countRef);
        unsigned callIndex = countRef++;

        result.states.push_back({state, callIndex, indexMap, valueMap, {}, {}});
        LLVM_DEBUG(llvm::dbgs() << "  visit " << state.getSymName() << " idx=" << callIndex
                                << "\n");

        IndexMap indexMapAfter = indexMap.advanced(accessesIt->second);
        auto resolved = resolveSuccessors(state, valueMapAfter);
        valueMapAfter.clearSnapshots();
        llvm::DenseSet<P4HIR::ParserStateOp> resolvedSet(resolved.begin(), resolved.end());

        for (auto successor : state.getNextStates()) {
            if (successor.isTerminal()) continue;
            if (resolvedSet.contains(successor))
                worklist.push_back({successor, indexMapAfter, valueMapAfter});
            else
                result.states.back().prunedSuccessors.insert(successor);
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "  symbolic execution: " << result.states.size()
                            << " state instances, " << result.visitedMap.size()
                            << " visited keys\n");
    return result;
}

}  // namespace P4::P4MLIR
