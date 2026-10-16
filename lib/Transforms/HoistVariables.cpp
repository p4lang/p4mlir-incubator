// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-hoist-variables"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_HOISTVARIABLES
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct HoistVariablesPass : public P4::P4MLIR::impl::HoistVariablesBase<HoistVariablesPass> {
    HoistVariablesPass() = default;
    void runOnOperation() override;
};

// Returns the region whose entry block hosts the variables declared at `op`: that of the nearest
// enclosing allocation scope that is not structured control flow, e.g. a function or a parser
// state. Values never cross operations isolated from above, and neither do variables.
Region *getAllocationRegion(Operation *op) {
    for (Region *region = op->getParentRegion(); region; region = region->getParentRegion()) {
        Operation *parentOp = region->getParentOp();
        if (parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) return region;
        if (parentOp->hasTrait<OpTrait::AutomaticAllocationScope>() &&
            !isa<RegionBranchOpInterface>(parentOp))
            return region;
    }
    return nullptr;
}

// Marks the lifetime of `variable` before it is hoisted: it starts where the variable is
// declared and ends at scope exits dominated by the declaration. At other exits, omitting the
// end marker conservatively keeps the storage live: the declaration may not have executed on
// every incoming path. Variables declared directly in an allocation scope remain live until
// that scope ends.
void markLifetime(OpBuilder &builder, P4HIR::VariableOp variable, DominanceInfo &dominance) {
    // Leave lifetimes that are already marked alone.
    if (llvm::any_of(variable->getUsers(),
                     llvm::IsaPred<P4HIR::LifetimeStartOp, P4HIR::LifetimeEndOp>))
        return;

    auto loc = variable.getLoc();
    builder.setInsertionPointAfter(variable);
    P4HIR::LifetimeStartOp::create(builder, loc, variable);

    Region *scope = variable->getParentRegion();
    if (!isa<RegionBranchOpInterface>(scope->getParentOp())) return;

    for (Block &block : *scope) {
        Operation *terminator = block.getTerminator();
        if (terminator->getNumSuccessors() ||
            !dominance.dominates(variable.getOperation(), terminator))
            continue;
        builder.setInsertionPoint(terminator);
        P4HIR::LifetimeEndOp::create(builder, loc, variable);
    }
}

void HoistVariablesPass::runOnOperation() {
    // Collect the variables first, so that moving them does not interfere with the walk.
    SmallVector<std::pair<P4HIR::VariableOp, Block *>> variables;
    getOperation()->walk([&](P4HIR::VariableOp op) {
        Region *region = getAllocationRegion(op);
        if (!region) return;
        Block *entryBlock = &region->front();
        if (op->getBlock() != entryBlock) variables.emplace_back(op, entryBlock);
    });

    // Moving every variable before the original first operation of its entry block keeps the
    // hoisted variables in the order of their declarations.
    OpBuilder builder(&getContext());
    DominanceInfo dominance(getOperation());
    DenseMap<Block *, Operation *> insertionPoints;
    for (auto [op, entryBlock] : variables) {
        markLifetime(builder, op, dominance);
        auto it = insertionPoints.try_emplace(entryBlock, &entryBlock->front()).first;
        op->moveBefore(it->second);
    }
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createHoistVariablesPass() {
    return std::make_unique<HoistVariablesPass>();
}
