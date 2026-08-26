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

/// Extensible helper class to walk field-accesses and their uses starting from some root value. The
/// default implementation supports walking struct field and array element with constant index
/// field-access operations.
class PathWalker {
 public:
    using IndirectUsesMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;
    using NodeCallbackFn = std::function<mlir::WalkResult(mlir::Value, P4HIR::FieldPath)>;
    using LeafCallbackFn =
        std::function<mlir::WalkResult(mlir::Operation *, mlir::OpOperand &, P4HIR::FieldPath)>;

    PathWalker() {}
    virtual ~PathWalker() {}

    /// Return an indirect use mapping for P4HIR `control_local` and `symbol_ref`-based accesses.
    static IndirectUsesMap getIndirectSymbolUses(P4HIR::ControlOp control) {
        IndirectUsesMap result;
        control.walk([&](P4HIR::SymToValueOp symbolRef) {
            if (auto referencedControlLocal = symbolRef.getDeclOp<P4HIR::ControlLocalOp>()) {
                auto [it, ins] = result.insert({referencedControlLocal.getVal(), {}});
                it->second.push_back(symbolRef.getResult());
            }
        });
        return result;
    }

    /// Set the indirect use mapping to use. This mapping provides additional uses for values that
    /// are accessed by means other than SSA def-use chains (e.g. symbol-based access).
    PathWalker &setIndirectUsesMap(const IndirectUsesMap *indirectUsesMap) {
        indirectUses = indirectUsesMap;
        return *this;
    }

    /// Set the callback function for leaf operations. This callback is called for operations that
    /// are not field-access operations and that access a value with a known path. The operation,
    /// the operand and the corresponding path are provided as arguments to the callback. If a leaf
    /// operation has multiple operands with known paths then the callback will be called multiple
    /// times.
    PathWalker &setLeafCallback(LeafCallbackFn cb) {
        leafCb = std::move(cb);
        return *this;
    }

    /// Set the callback function for intermediate nodes. This callback is called for field-access
    /// operations. The result value and the corresponding accessed field are provided as arguments
    /// to the callback.
    PathWalker &setNodeCallback(NodeCallbackFn cb) {
        nodeCb = std::move(cb);
        return *this;
    }

    /// Helper to look through a P4HIR `struct_extract` or `struct_field_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> lookThroughStructAccess(
        mlir::Operation *op, mlir::OpOperand &operand, P4HIR::FieldPath path) {
        return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
            .Case<P4HIR::StructExtractOp, P4HIR::StructFieldRefOp>([&](auto structAccessOp) {
                assert((operand.get() == structAccessOp.getInput()) && "Unexpected operand");
                return std::pair{structAccessOp.getResult(), path[structAccessOp.getFieldIndex()]};
            })
            .Default(
                [](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
    }

    /// Helper to look through a P4HIR `array_get` or `array_element_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> lookThroughArrayAccess(mlir::Operation *op,
                                                                           mlir::OpOperand &operand,
                                                                           P4HIR::FieldPath path) {
        return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
            .Case<P4HIR::ArrayGetOp, P4HIR::ArrayElementRefOp>(
                [&](auto arrayAccessOp) -> std::pair<mlir::Value, P4HIR::FieldPath> {
                    assert((operand.get() == arrayAccessOp.getInput()) && "Unexpected operand");

                    unsigned cstIndex;
                    if (matchPattern(arrayAccessOp.getIndex(), m_ConstantInt(&cstIndex)))
                        return {arrayAccessOp.getResult(), path[cstIndex]};

                    return {mlir::Value(), P4HIR::FieldPath()};
                })
            .Default(
                [](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
    }

    /// `operand` is an operand in `op` that is used to access the root value's field that
    /// corresponds to `path`. If `op` is a field access operation and the resulting path can be
    /// determined based on `path`, then this function should return the resulting new value and
    /// path. Otherwise an empty value must be returned.
    virtual std::pair<mlir::Value, P4HIR::FieldPath> lookThroughUser(mlir::Operation *op,
                                                                     mlir::OpOperand &operand,
                                                                     P4HIR::FieldPath path) {
        // Default implementation to handle P4HIR operations. Users should override this to also
        // support operations from other dialects.
        auto lookThroughStruct = lookThroughStructAccess(op, operand, path);
        if (lookThroughStruct.first) return lookThroughStruct;

        auto lookThroughArray = lookThroughArrayAccess(op, operand, path);
        if (lookThroughArray.first) return lookThroughStruct;

        return {mlir::Value(), P4HIR::FieldPath()};
    }

    /// Walk field-access operations and uses starting from `root`. If `rootType` is additionally
    /// provided it is used to compute field paths instead of `root.getType()`. This is useful when
    /// doing type convertions. If `rootType` is provided it must have the same shape as
    /// `root.getType()`.
    mlir::WalkResult walk(mlir::Value root, mlir::Type rootType = mlir::Type()) {
        if (!rootType) rootType = root.getType();

        P4HIR::FieldPath rootPath(P4HIR::maybeUnref(rootType));
        return walkImpl(root, rootPath);
    }

 protected:
    // Value can only be `root` or otherwise a result of a field access operation.
    mlir::WalkResult walkImpl(mlir::Value value, P4HIR::FieldPath path) {
        auto status = nodeCb(value, path);
        if (status.wasSkipped()) return mlir::WalkResult::advance();
        if (status.wasInterrupted()) return mlir::WalkResult::interrupt();

        for (auto &use : value.getUses()) {
            mlir::Operation *user = use.getOwner();

            auto [resValue, resPath] = lookThroughUser(user, use, path);
            if (resValue) {
                if (walkImpl(resValue, resPath).wasInterrupted())
                    return mlir::WalkResult::interrupt();
            } else {
                if (leafCb(user, use, path).wasInterrupted()) return mlir::WalkResult::interrupt();
            }
        }

        if (indirectUses) {
            auto it = indirectUses->find(value);
            if (it != indirectUses->end()) {
                for (mlir::Value alias : it->second)
                    if (walkImpl(alias, path).wasInterrupted())
                        return mlir::WalkResult::interrupt();
            }
        }

        return mlir::WalkResult::advance();
    }

    const IndirectUsesMap *indirectUses = nullptr;
    NodeCallbackFn nodeCb;
    LeafCallbackFn leafCb;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
