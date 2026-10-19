// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"
#include "p4mlir/Dialect/P4HIR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::IRUtils {

/// Assuming that `args` are subset of existing arguments of `obj`, duplicate each one of `args`.
/// Each duplicated argument is placed just before the original and has the same attributes
/// and location. If `newTypes` is non-empty then these types are used for the duplicated
/// arguments, otherwise the original types are used.
mlir::FailureOr<llvm::SmallVector<mlir::BlockArgument>> duplicateFunctionLikeArgs(
    mlir::FunctionOpInterface obj, llvm::ArrayRef<mlir::BlockArgument> args,
    llvm::ArrayRef<mlir::Type> newTypes = {});

/// Assuming that `args` are subset of existing arguments of `obj`, erase all these arguments from
/// `obj`.
mlir::LogicalResult eraseFunctionLikeArgs(mlir::FunctionOpInterface obj,
                                          llvm::ArrayRef<mlir::BlockArgument> args);

/// Legalize `str` to a name that only contains alphanumeric characters and underscores.
std::string legalizeName(std::string str);

/// Create a struct type named `name` and with fields based on `paths`.
P4HIR::StructType createStructType(mlir::MLIRContext *ctx, const llvm::Twine &name,
                                   const llvm::SmallVectorImpl<P4HIR::FieldPath> &paths,
                                   mlir::DictionaryAttr annotations = {});

/// Make `val` accessible to `op` regardless of wherher `op` is inside an action.
mlir::Value getAccessibleValue(mlir::RewriterBase &rewriter, mlir::Operation *op, mlir::Value val);

// Inline `scopeOp`'s body to its parent.
void inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp);

// If `op` is an operation somewhere within `block`, check if we can split it in three parts:
// One with operations before `op`, one with `op` and one with operations after `op`.
// This function should be called to check if it's possible to use `splitBlockAt`.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split block in three parts: One with operations before `op`, one with `op` and one with
// operations after `op`. Any scopes surrounding `op` may be inlined to perform the split.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    const llvm::Twine &suffix);

// Helper class to replace one operation in a state with multiple new states.
// During init it creates two "pre" and "post" states in which the code before and after the split
// point is put. Then any number of additional states between pre and post can be created with the
// createXXXState functions. Once done the user should either call finalize to commit changes or
// cancel and any changes will be undone.
class SplitStateRewriter {
 public:
    SplitStateRewriter(mlir::RewriterBase &rewriter, mlir::Operation *op)
        : rewriter(rewriter), op(op), step(CREATED) {}
    SplitStateRewriter(SplitStateRewriter &&) = delete;
    SplitStateRewriter &operator=(SplitStateRewriter &&) = delete;
    SplitStateRewriter(const SplitStateRewriter &) = delete;
    SplitStateRewriter &operator=(const SplitStateRewriter &) = delete;
    ~SplitStateRewriter() {
        assert((step == CANCELED || step == FINALIZED) && "Incorrect state on destruction");
    }

    // Initialize and check if splitting is feasible.
    mlir::LogicalResult init();

    // Create new intermediate state that transitions to `transitionTo` and move `ops` in it.
    // If `ops` is non-null the terminator of the block is erased once moved.
    P4HIR::ParserStateOp createSubState(const llvm::Twine &suffix,
                                        P4HIR::ParserStateOp transitionTo = {},
                                        mlir::Block *ops = nullptr);

    // Same as createSubState but transition to the "post" state specifically.
    P4HIR::ParserStateOp createJoinSubState(const llvm::Twine &suffix, mlir::Block *ops = nullptr) {
        return createSubState(suffix, postState, ops);
    }

    // Undo any changes done so far.
    void cancel();

    // Finalize and commit all changes.
    void finalize();

    mlir::Operation *getSplitOp() { return op; }
    P4HIR::ParserStateOp getState() { return op->getParentOfType<P4HIR::ParserStateOp>(); }

    P4HIR::ParserStateOp getPreState() {
        assert(step == INITIALIZED);
        return preState;
    }

    P4HIR::ParserStateOp getPostState() {
        assert(step == INITIALIZED);
        return postState;
    }

    void setStateInsertionPointAfter(P4HIR::ParserStateOp afterState) {
        stateCreationPoint = afterState;
    }

 private:
    enum RewriteStep { CREATED, INITIALIZED, CANCELED, FINALIZED };

    mlir::RewriterBase &rewriter;
    mlir::Operation *op;
    RewriteStep step;

    P4HIR::ParserStateOp preState;
    P4HIR::ParserStateOp postState;
    mlir::Operation *stateCreationPoint;
};

/// Helper class that efficiently provides additional indirect uses for values. Indirect uses are
/// uses that are made by means other than SSA def-use chains (e.g. symbol-based access).
class IndirectUses {
 public:
    using MapType = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;

    /// Returns indirect uses of `val`.
    llvm::ArrayRef<mlir::Value> getIndirectUses(mlir::Value val) {
        mlir::Operation *owner;
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val))
            owner = blockArg.getOwner()->getParentOp();
        else
            owner = val.getDefiningOp();

        if (!owner) return {};

        auto *mapping = getMapImpl(owner);
        if (!mapping) return {};

        auto it = mapping->find(val);
        if (it == mapping->end()) return {};

        return it->second;
    }

 protected:
    /// Populate `mapping` to acount for `control_local` and `symbol_ref`-based accesses in `owner`.
    static void populateControlLocalIndirectUses(mlir::Operation *owner, MapType &mapping);

    MapType *getMapImpl(mlir::Operation *owner);

    llvm::DenseMap<mlir::Operation *, std::unique_ptr<MapType>> mappings;
};

/// Helper class containing common FieldPath related functions that are used in other
/// transformations and utilities.
class P4HIRFieldPathOps {
 protected:
    /// Helper to look through a P4HIR `struct_extract` or `struct_field_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> getAccessedFieldStruct(
        mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path);

    /// Helper to look through a P4HIR `array_get` or `array_element_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> getAccessedFieldArray(
        mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path);

 public:
    /// `operand` is an operand in `op` that corresponds to `path`. If `op` is a field access
    /// operation and the resulting path can be determined, this function returns the resulting
    /// field access value and path. Otherwise an empty value is returned.
    static std::pair<mlir::Value, P4HIR::FieldPath> getAccessedField(mlir::Operation *op,
                                                                     const mlir::OpOperand &operand,
                                                                     P4HIR::FieldPath path);

    /// Builds operations to access the field that corresponds to `field` from `value`.
    static mlir::Value buildFieldAccess(mlir::RewriterBase &rewriter, mlir::Location loc,
                                        mlir::Value val, P4HIR::IndexedField field);

    /// Builds operations to access the field that corresponds to `path` from `value`.
    static mlir::Value buildPathAccess(mlir::RewriterBase &rewriter, mlir::Location loc,
                                       mlir::Value val, P4HIR::FieldPath path) {
        for (auto field : path.fields()) val = buildFieldAccess(rewriter, loc, val, field);
        return val;
    }
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
