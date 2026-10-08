// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-expand-slice-read-assign"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_EXPANDSLICEREADASSIGN
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
// Builds the value of the object referenced by `op` after a slice assignment, replacing
// bits [`lowBit`, `highBit`] in its previous value `old`.
Value buildAssignedValue(OpBuilder &builder, P4HIR::AssignSliceOp op, Value old) {
    auto objectType = llvm::cast<P4HIR::BitsType>(
        llvm::cast<P4HIR::ReferenceType>(op.getRef().getType()).getObjectType());
    unsigned width = objectType.getWidth();
    unsigned hi = op.getHighBit(), lo = op.getLowBit();
    Location loc = op.getLoc();

    // Like p4c's RemoveLeftSlices: (old & ~mask) | (value << lowBit), where `mask` covers the
    // slice. The value is unsigned: casting it to the type of the object zero-extends it, so no
    // bits outside the slice are set after the shift.
    Value value = op.getValue();
    if (value.getType() != objectType)
        value = P4HIR::CastOp::create(builder, loc, objectType, value);
    if (lo > 0) {
        auto shiftType = P4HIR::BitsType::get(builder.getContext(), width, false);
        Value shift = P4HIR::ConstOp::create(builder, loc, P4HIR::IntAttr::get(shiftType, lo));
        value = P4HIR::ShlOp::create(builder, loc, value, shift);
    }

    if (!old) return value;

    llvm::APInt keepBits = ~llvm::APInt::getBitsSet(width, lo, hi + 1);
    Value keep = P4HIR::ConstOp::create(builder, loc, P4HIR::IntAttr::get(objectType, keepBits));
    Value kept = P4HIR::BinOp::create(builder, loc, P4HIR::BinOpKind::And, old, keep);
    return P4HIR::BinOp::create(builder, loc, P4HIR::BinOpKind::Or, kept, value);
}

struct ExpandSliceReadAssignPass
    : public P4::P4MLIR::impl::ExpandSliceReadAssignBase<ExpandSliceReadAssignPass> {
    ExpandSliceReadAssignPass() = default;
    void runOnOperation() override;
};

void ExpandSliceReadAssignPass::runOnOperation() {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](Operation *op) {
        bool expand = (expandReadSlice && isa<P4HIR::ReadSliceOp>(op)) ||
                      (expandAssignSlice && isa<P4HIR::AssignSliceOp>(op));
        if (!expand) return;
        rewriter.setInsertionPoint(op);
        if (auto read = llvm::dyn_cast<P4HIR::ReadSliceOp>(op)) {
            // Read the whole object and take the slice of its value.
            Value value = P4HIR::ReadOp::create(rewriter, read.getLoc(), read.getInput());
            rewriter.replaceOpWithNewOp<P4HIR::SliceOp>(read, read.getType(), value,
                                                        read.getHighBit(), read.getLowBit());
        } else {
            auto assign = llvm::cast<P4HIR::AssignSliceOp>(op);
            // Read the whole object, replace the slice in its value, and write it back.
            // A slice covering the whole object leaves no bits to keep: no read is needed.
            Value old = assign.coversWholeObject()
                            ? Value()
                            : P4HIR::ReadOp::create(rewriter, assign.getLoc(), assign.getRef());
            Value updated = buildAssignedValue(rewriter, assign, old);
            rewriter.replaceOpWithNewOp<P4HIR::AssignOp>(assign, updated, assign.getRef());
        }
    });
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createExpandSliceReadAssignPass() {
    return std::make_unique<ExpandSliceReadAssignPass>();
}
