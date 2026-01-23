#include "llvm/Support/ErrorHandling.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

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
