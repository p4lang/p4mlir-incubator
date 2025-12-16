//===----------------------------------------------------------------------===//
//
// This file implements an algorithm for eliminating symbol operations that are
// known to be dead. This is essentially SymbolDCE from MLIR mainline but with
// P4HIR specifics (top-level symbol references)
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "p4hir-symbol-dce"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SYMBOLDCE
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {
struct SymbolDCE : public impl::SymbolDCEBase<SymbolDCE> {
    void runOnOperation() override;

    /// Compute the liveness of the symbols within the given symbol table.
    /// `symbolTableIsHidden` is true if this symbol table is known to be
    /// unaccessible from operations in its parent regions.
    LogicalResult computeLiveness(Operation *symbolTableOp, SymbolTableCollection &symbolTable,
                                  bool symbolTableIsHidden, DenseSet<Operation *> &liveSymbols);
};
}  // namespace

void SymbolDCE::runOnOperation() {
    Operation *symbolTableOp = getOperation();

    // SymbolDCE should only be run on operations that define a symbol table.
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
        symbolTableOp->emitOpError()
            << " was scheduled to run under SymbolDCE, but does not define a "
               "symbol table";
        return signalPassFailure();
    }

    // A flag that signals if the top level symbol table is hidden, i.e. not
    // accessible from parent scopes.
    bool symbolTableIsHidden = true;
    SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(symbolTableOp);
    if (symbolTableOp->getParentOp() && symbol) symbolTableIsHidden = symbol.isPrivate();

    // Compute the set of live symbols within the symbol table.
    DenseSet<Operation *> liveSymbols;
    SymbolTableCollection symbolTable;
    if (failed(computeLiveness(symbolTableOp, symbolTable, symbolTableIsHidden, liveSymbols)))
        return signalPassFailure();

    // After computing the liveness, delete all of the symbols that were found to
    // be dead.
    symbolTableOp->walk([&](Operation *nestedSymbolTable) {
        if (!nestedSymbolTable->hasTrait<OpTrait::SymbolTable>()) return;
        bool emptyOp = true;
        for (auto &block : nestedSymbolTable->getRegion(0)) {
            for (Operation &op : llvm::make_early_inc_range(block)) {
                if (isa<SymbolOpInterface>(&op) && !liveSymbols.count(&op)) {
                    LLVM_DEBUG(llvm::dbgs() << "erasing: " << op << "\n");
                    op.erase();
                    ++numDCE;
                }
            }
            emptyOp &= block.empty();
        }
        if (emptyOp && isa<SymbolOpInterface>(nestedSymbolTable) &&
            !liveSymbols.count(nestedSymbolTable)) {
            LLVM_DEBUG(llvm::dbgs() << "erasing: " << *nestedSymbolTable << "\n");
            nestedSymbolTable->erase();
            ++numDCE;
        }
    });
}

/// Compute the liveness of the symbols within the given symbol table.
/// `symbolTableIsHidden` is true if this symbol table is known to be
/// unaccessible from operations in its parent regions.
LogicalResult SymbolDCE::computeLiveness(Operation *symbolTableOp,
                                         SymbolTableCollection &symbolTable,
                                         bool symbolTableIsHidden,
                                         DenseSet<Operation *> &liveSymbols) {
    LLVM_DEBUG(llvm::dbgs() << "computeLiveness: " << symbolTableOp->getName() << " ("
                            << symbolTableOp << ")\n");
    // A worklist of live operations to propagate uses from.
    SmallVector<Operation *, 16> worklist;

    // Walk the symbols within the current symbol table, marking the symbols that
    // are known to be live.
    for (auto &block : symbolTableOp->getRegion(0)) {
        // Add all non-symbols or symbols that can't be discarded.
        for (Operation &op : block) {
            SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(&op);
            if (!symbol) {
                LLVM_DEBUG(llvm::dbgs() << " adding " << op.getName() << " to worklist\n");
                worklist.push_back(&op);
                continue;
            }
            bool isDiscardable =
                (symbolTableIsHidden || symbol.isPrivate()) && symbol.canDiscardOnUseEmpty();
            LLVM_DEBUG(llvm::dbgs()
                       << " symbol @" << symbol.getName()
                       << (isDiscardable ? " is discardable\n" : " is not discardable\n"));
            if (!isDiscardable && liveSymbols.insert(&op).second) {
                LLVM_DEBUG(llvm::dbgs()
                           << "  marking symbol @" << symbol.getName() << " as live\n");
                worklist.push_back(&op);
            }
        }
    }

    // Process the set of symbols that were known to be live, adding new symbols
    // that are referenced within. For operations that are not symbol tables, it
    // considers the liveness with respect to the op itself rather than scope of
    // nested symbol tables by enqueuing all the top level operations for
    // consideration.
    while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op);

        LLVM_DEBUG(llvm::dbgs() << " processing: " << op->getName();
                   if (symbol) llvm::dbgs() << " @" << symbol.getName(); llvm::dbgs() << "\n");

        // If this is a symbol table, recursively compute its liveness.
        if (op->hasTrait<OpTrait::SymbolTable>()) {
            // The internal symbol table is hidden if the parent is, if its not a
            // symbol, or if it is a private symbol.
            bool symIsHidden = symbolTableIsHidden || !symbol || symbol.isPrivate();
            LLVM_DEBUG(llvm::dbgs() << " symbol table: " << op->getName() << " is "
                                    << (symIsHidden ? "hidden" : "public") << "\n");
            if (failed(computeLiveness(op, symbolTable, symIsHidden, liveSymbols)))
                return failure();
        } else {
            LLVM_DEBUG(llvm::dbgs() << " non-symbol table: " << op->getName() << "\n");
            // If the op is not a symbol table, then, unless op itself is dead which
            // would be handled by DCE, we need to check all the regions and blocks
            // within the op to find the uses (e.g., consider visibility within op as
            // if top level rather than relying on pure symbol table visibility). This
            // is more conservative than SymbolTable::walkSymbolTables in the case
            // where there is again SymbolTable information to take advantage of.
            for (auto &region : op->getRegions())
                for (auto &block : region.getBlocks())
                    for (Operation &op : block)
                        if (op.getNumRegions()) worklist.push_back(&op);
        }

        // Get the first parent symbol table op. Note: due to enqueueing of
        // top-level ops, we may not have a symbol table parent here, but if we do
        // not, then we also don't have a symbol.
        Operation *parentOp = op->getParentOp();
        if (!parentOp->hasTrait<OpTrait::SymbolTable>()) continue;
        Operation *moduleOp = op->getParentOfType<mlir::ModuleOp>();
        if (!moduleOp) continue;

        // Collect the uses held by this operation.
        std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(op);
        if (!uses) {
            return op->emitError()
                   << "operation contains potentially unknown symbol table, meaning "
                   << "that we can't reliable compute symbol uses";
        }

        if (uses->empty()) continue;

        LLVM_DEBUG(llvm::dbgs() << " uses nested in " << op->getName() << "\n");
        SmallVector<Operation *, 4> resolvedSymbols;
        for (const SymbolTable::SymbolUse &use : *uses) {
            LLVM_DEBUG(llvm::dbgs() << "  use: " << use.getSymbolRef() << "\n");
            // Lookup the symbols referenced by this use.
            resolvedSymbols.clear();
            if (failed(symbolTable.lookupSymbolIn(parentOp, use.getSymbolRef(), resolvedSymbols)) &&
                failed(symbolTable.lookupSymbolIn(moduleOp, use.getSymbolRef(), resolvedSymbols))) {
                LLVM_DEBUG(llvm::dbgs() << "   unknown symbol, ignoring\n");
                // Ignore references to unknown symbols.
                continue;
            }

            LLVM_DEBUG({
                llvm::dbgs() << "   resolved symbols: ";
                llvm::interleaveComma(resolvedSymbols, llvm::dbgs(), [](Operation *op) {
                    SymbolOpInterface symbol = cast<SymbolOpInterface>(op);
                    llvm::dbgs() << op->getName() << " @" << symbol.getName();
                });
                llvm::dbgs() << "\n";
            });

            // Mark each of the resolved symbols as live.
            for (Operation *resolvedSymbol : resolvedSymbols)
                if (liveSymbols.insert(resolvedSymbol).second) worklist.push_back(resolvedSymbol);
        }
    }

    return success();
}

std::unique_ptr<Pass> createSymbolDCEPass() { return std::make_unique<SymbolDCE>(); }

}  // namespace P4::P4MLIR
