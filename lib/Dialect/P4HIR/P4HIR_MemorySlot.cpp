#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> P4HIR::VariableOp::getPromotableSlots() {
    return {MemorySlot{getResult(), getObjectType()}};
}

Value P4HIR::VariableOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
    if (auto defaultValueType = mlir::dyn_cast<HasDefaultValue>(slot.elemType))
        return builder.create<P4HIR::ConstOp>(getLoc(), defaultValueType.getDefaultValue());

    // TODO: This should really not happen
    llvm_unreachable("cannot materialize default value");
    return builder.create<P4HIR::UninitializedOp>(getLoc(), slot.elemType);
}

void P4HIR::VariableOp::handleBlockArgument(const MemorySlot &slot, BlockArgument argument,
                                            OpBuilder &builder) {}

std::optional<PromotableAllocationOpInterface> P4HIR::VariableOp::handlePromotionComplete(
    const MemorySlot &slot, Value defaultValue, OpBuilder &builder) {
    if (defaultValue && defaultValue.use_empty()) defaultValue.getDefiningOp()->erase();
    this->erase();
    return std::nullopt;
}

SmallVector<DestructurableMemorySlot> P4HIR::VariableOp::getDestructurableSlots() {
    auto destructurable = llvm::dyn_cast<DestructurableTypeInterface>(getObjectType());
    if (!destructurable) return {};

    std::optional<DenseMap<Attribute, Type>> destructuredType =
        destructurable.getSubelementIndexMap();
    if (!destructuredType) return {};

    return {DestructurableMemorySlot{{getResult(), getObjectType()}, *destructuredType}};
}

DenseMap<Attribute, MemorySlot> P4HIR::VariableOp::destructure(
    const DestructurableMemorySlot &slot, const SmallPtrSetImpl<Attribute> &usedIndices,
    OpBuilder &builder, SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
    builder.setInsertionPointAfter(*this);

    DenseMap<Attribute, MemorySlot> slotMap;
    SmallVector<Attribute> sortedIndices(usedIndices.begin(), usedIndices.end());
    llvm::sort(sortedIndices, [](auto a, auto b) {
        return cast<IntegerAttr>(a).getInt() < cast<IntegerAttr>(b).getInt();
    });

    auto objectType = llvm::cast<DestructurableTypeInterface>(getObjectType());
    for (Attribute usedIndex : sortedIndices) {
        Type elemType = objectType.getTypeAtIndex(usedIndex);
        assert(elemType && "used index must exist");
        ReferenceType elemRef = P4HIR::ReferenceType::get(elemType);
        auto optName = getName();
        // TODO: Switch to FieldInfo, so we can deduce proper field name
        auto subVariable = builder.create<P4HIR::VariableOp>(
            getLoc(), elemRef,
            optName ? *optName + ".field" + Twine(cast<IntegerAttr>(usedIndex).getInt()) : "",
            getInit(), getAnnotationsAttr());
        newAllocators.push_back(subVariable);
        slotMap.try_emplace<MemorySlot>(usedIndex, {subVariable.getResult(), elemType});
    }

    return slotMap;
}

std::optional<DestructurableAllocationOpInterface> P4HIR::VariableOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder &builder) {
    assert(slot.ptr == getResult());
    this->erase();
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Interfaces for ReadOp
//===----------------------------------------------------------------------===//

bool P4HIR::ReadOp::loadsFrom(const MemorySlot &slot) { return getRef() == slot.ptr; }

bool P4HIR::ReadOp::storesTo(const MemorySlot &slot) { return false; }

Value P4HIR::ReadOp::getStored(const MemorySlot &slot, OpBuilder &builder, Value reachingDef,
                               const DataLayout &dataLayout) {
    llvm_unreachable("getStored should not be called on ReadOp");
}

bool P4HIR::ReadOp::canUsesBeRemoved(const MemorySlot &slot,
                                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                     SmallVectorImpl<OpOperand *> &newBlockingUses,
                                     const DataLayout &dataLayout) {
    if (blockingUses.size() != 1) return false;
    Value blockingUse = (*blockingUses.begin())->get();
    return blockingUse == slot.ptr && getRef() == slot.ptr && getType() == slot.elemType;
}

DeletionKind P4HIR::ReadOp::removeBlockingUses(const MemorySlot &slot,
                                               const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                               OpBuilder &builder, Value reachingDefinition,
                                               const DataLayout &dataLayout) {
    getResult().replaceAllUsesWith(reachingDefinition);
    return DeletionKind::Delete;
}

bool P4HIR::ReadOp::canRewire(const DestructurableMemorySlot &slot,
                              SmallPtrSetImpl<Attribute> &usedIndices,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
    if (slot.ptr != getRef()) return false;

    for (auto [idx, _] : slot.subelementTypes) usedIndices.insert(idx);

    return true;
}

static void getSortedPtrs(DenseMap<Attribute, MemorySlot> &subslots,
                          SmallVectorImpl<std::pair<unsigned, Value>> &sorted) {
    for (auto [attr, mem] : subslots) {
        assert(isa<IntegerAttr>(attr));
        sorted.emplace_back(cast<IntegerAttr>(attr).getInt(), mem.ptr);
    }

    llvm::sort(sorted, [](auto a, auto b) { return a.first < b.first; });
}

DeletionKind P4HIR::ReadOp::rewire(const DestructurableMemorySlot &slot,
                                   DenseMap<Attribute, MemorySlot> &subslots, OpBuilder &builder,
                                   const DataLayout &dataLayout) {
    SmallVector<std::pair<unsigned, mlir::Value>> elements;
    getSortedPtrs(subslots, elements);

    SmallVector<mlir::Value> vals;
    for (auto [_, val] : elements) vals.push_back(builder.create<P4HIR::ReadOp>(getLoc(), val));

    Value repl = mlir::TypeSwitch<Type, Value>(getType())
                     .Case<P4HIR::StructType>([&](auto) {
                         return builder.create<P4HIR::StructOp>(getLoc(), slot.elemType, vals);
                     })
                     .Case<P4HIR::ArrayType>([&](auto) {
                         return builder.create<P4HIR::ArrayOp>(getLoc(), slot.elemType, vals);
                     });

    replaceAllUsesWith(repl);
    return DeletionKind::Delete;
}

LogicalResult P4HIR::ReadOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                                    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                                    const DataLayout &dataLayout) {
    return success();
}

//===----------------------------------------------------------------------===//
// Interfaces for AssignOp
//===----------------------------------------------------------------------===//

bool P4HIR::AssignOp::loadsFrom(const MemorySlot &slot) { return false; }

bool P4HIR::AssignOp::storesTo(const MemorySlot &slot) { return getRef() == slot.ptr; }

Value P4HIR::AssignOp::getStored(const MemorySlot &slot, OpBuilder &builder, Value reachingDef,
                                 const DataLayout &dataLayout) {
    return getValue();
}

bool P4HIR::AssignOp::canUsesBeRemoved(const MemorySlot &slot,
                                       const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                       SmallVectorImpl<OpOperand *> &newBlockingUses,
                                       const DataLayout &dataLayout) {
    if (blockingUses.size() != 1) return false;
    Value blockingUse = (*blockingUses.begin())->get();
    return blockingUse == slot.ptr && getRef() == slot.ptr && getValue() != slot.ptr &&
           slot.elemType == getValue().getType();
}

DeletionKind P4HIR::AssignOp::removeBlockingUses(const MemorySlot &slot,
                                                 const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                                 OpBuilder &builder, Value reachingDefinition,
                                                 const DataLayout &dataLayout) {
    return DeletionKind::Delete;
}

bool P4HIR::AssignOp::canRewire(const DestructurableMemorySlot &slot,
                                SmallPtrSetImpl<Attribute> &usedIndices,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const DataLayout &dataLayout) {
    if (slot.ptr != getRef()) return false;
    for (auto [idx, _] : slot.subelementTypes) usedIndices.insert(idx);

    return true;
}

DeletionKind P4HIR::AssignOp::rewire(const DestructurableMemorySlot &slot,
                                     DenseMap<Attribute, MemorySlot> &subslots, OpBuilder &builder,
                                     const DataLayout &dataLayout) {
    SmallVector<std::pair<unsigned, mlir::Value>> elements;
    getSortedPtrs(subslots, elements);

    mlir::TypeSwitch<Type>(slot.elemType)
        .Case<StructLikeTypeInterface>([&](auto structType) {
            for (auto [idx, elt] : elements) {
                auto &fieldInfo = structType.getFields()[idx];
                auto val = builder.create<P4HIR::StructExtractOp>(getLoc(), getValue(), fieldInfo);
                builder.create<P4HIR::AssignOp>(getLoc(), val, elt);
            }
        })
        .Case<ArrayType>([&](auto arrayType) {
            for (auto [idx, elt] : elements) {
                auto idxConst = builder.create<P4HIR::ConstOp>(
                    getLoc(),
                    P4HIR::IntAttr::get(P4HIR::BitsType::get(getContext(), 32, false), idx));
                auto val = builder.create<P4HIR::ArrayGetOp>(getLoc(), getValue(), idxConst);
                builder.create<P4HIR::AssignOp>(getLoc(), val, elt);
            }
        });

    return DeletionKind::Delete;
}

LogicalResult P4HIR::AssignOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                                      SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                                      const DataLayout &dataLayout) {
    return success();
}
//===----------------------------------------------------------------------===//
// Interfaces for StructFieldRefOp
//===----------------------------------------------------------------------===//

bool P4HIR::StructFieldRefOp::canRewire(const DestructurableMemorySlot &slot,
                                        SmallPtrSetImpl<Attribute> &usedIndices,
                                        SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                        const DataLayout &dataLayout) {
    if (slot.ptr != getInput()) return false;

    auto indexAttr = IntegerAttr::get(IndexType::get(getContext()), getFieldIndex());

    if (!slot.subelementTypes.contains(indexAttr)) return false;

    usedIndices.insert(indexAttr);
    mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), getResult().getType().getObjectType()});

    return true;
}

DeletionKind P4HIR::StructFieldRefOp::rewire(const DestructurableMemorySlot &slot,
                                             DenseMap<Attribute, MemorySlot> &subslots,
                                             OpBuilder &builder, const DataLayout &dataLayout) {
    auto indexAttr = IntegerAttr::get(IndexType::get(getContext()), getFieldIndex());

    auto it = subslots.find(indexAttr);
    assert(it != subslots.end());

    replaceAllUsesWith(it->getSecond().ptr);

    return DeletionKind::Delete;
}

LogicalResult P4HIR::StructFieldRefOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
    return success();
}

//===----------------------------------------------------------------------===//
// Interfaces for ArrayElementRefOp
//===----------------------------------------------------------------------===//

// TODO: Ensure access is in-bounds
LogicalResult P4HIR::ArrayElementRefOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
    return success();
}

bool P4HIR::ArrayElementRefOp::canRewire(const DestructurableMemorySlot &slot,
                                         SmallPtrSetImpl<Attribute> &usedIndices,
                                         SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                         const DataLayout &dataLayout) {
    if (slot.ptr != getInput()) return false;

    // Can only rewire constant indices
    auto cstIndex = getIndex().getDefiningOp<P4HIR::ConstOp>();
    if (!cstIndex) return false;

    auto indexAttr = IntegerAttr::get(IndexType::get(getContext()),
                                      cstIndex.getValueAs<P4HIR::IntAttr>().getUInt());

    if (!slot.subelementTypes.contains(indexAttr)) return false;

    usedIndices.insert(indexAttr);
    mustBeSafelyUsed.emplace_back<MemorySlot>({getResult(), getResult().getType().getObjectType()});

    return true;
}

DeletionKind P4HIR::ArrayElementRefOp::rewire(const DestructurableMemorySlot &slot,
                                              DenseMap<Attribute, MemorySlot> &subslots,
                                              OpBuilder &builder, const DataLayout &dataLayout) {
    auto cstIndex = cast<P4HIR::ConstOp>(getIndex().getDefiningOp());
    auto indexAttr = IntegerAttr::get(IndexType::get(getContext()),
                                      cstIndex.getValueAs<P4HIR::IntAttr>().getUInt());

    auto it = subslots.find(indexAttr);
    assert(it != subslots.end());

    replaceAllUsesWith(it->getSecond().ptr);

    return DeletionKind::Delete;
}
