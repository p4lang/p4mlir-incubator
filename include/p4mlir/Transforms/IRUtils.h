#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/PatternMatch.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::IRUtils {

// Inline `scopeOp`'s body to its parent.
void inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp);

// Return true if it's valid to call `splitBlockAt` for the given arguments.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split `block` that has as ancestor `op` in three:
// One with operations before `op`, one with `op` and one with operations after `op`.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    llvm::StringRef suffix);

};  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
