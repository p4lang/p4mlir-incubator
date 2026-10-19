// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_PATH_REWRITER_H
#define P4MLIR_IMPL_UTILS_PATH_REWRITER_H

#include "p4mlir/Utilities/IRUtils.h"
#include "p4mlir/Utilities/PathWalker.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to replace uses of indexable-typed values with new values that may have
/// different layouts, using P4HIR::FieldPath. Users of an indexable-typed value may be
/// field-access operations, which produce new FieldPaths based on their operands, or leaf
/// operations which consume values.
/// The root value may either be a reference or a value of an indexable-type and the values to
/// replace are references or values accordingly.
template <class FieldPathOpsT>
class PathRewriter {
 public:
    using PathWalkerT = PathWalker<FieldPathOpsT>;

    /// Full constructor.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value root, mlir::Type type,
                 mlir::Value newRoot, mlir::Type newType)
        : rewriter(rewriter), root(root), newRoot(newRoot) {
        rootType = P4HIR::maybeUnref(type);
        if (newType) newRootType = P4HIR::maybeUnref(newType);
    }

    /// Minimal constructor.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue)
        : PathRewriter(rewriter, rootValue, rootValue.getType(), mlir::Value(), mlir::Type()) {}

    /// Constructor for value replacement.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue, mlir::Value newRootValue)
        : PathRewriter(rewriter, rootValue, rootValue.getType(), newRootValue,
                       newRootValue.getType()) {}

    PathRewriter(PathRewriter &&) = delete;
    PathRewriter &operator=(PathRewriter &&) = delete;
    PathRewriter(const PathRewriter &) = delete;
    PathRewriter &operator=(const PathRewriter &) = delete;
    virtual ~PathRewriter() {}

    /// `op` is an operation that has at least one operand accessing root value's fields. These
    /// operands and their paths can be found by using `getPath` on `op`'s arguments. If it is
    /// desirable to transform `op` then this should be done here. Return mlir::success() to
    /// continue the rewriting or mlir::failure() to interrupt it.
    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) { return mlir::success(); }

    /// `value` is a value used to access the root value's field that corresponds `path`. If it is
    /// desirable to replace `value` with a new equivalent, then this new value should be returned.
    /// If nothing should be changed then mlir::Value() should be returned.
    virtual mlir::Value replaceValue(mlir::Value value, P4HIR::FieldPath path) {
        return mlir::Value();
    }

    /// If `value` is accessing a field of the root value then return the corresponding field path,
    /// otherwise return the empty path.
    P4HIR::FieldPath getPath(mlir::Value value) {
        auto it = valueToPath.find(value);
        return it != valueToPath.end() ? it->second : P4HIR::FieldPath();
    }

    /// Assuming that `op` can only have a single operand with a corresponding field path, return
    /// that operand and path.
    std::pair<mlir::OpOperand *, P4HIR::FieldPath> getOperandWithPath(mlir::Operation *op) {
        P4HIR::FieldPath resPath;
        mlir::OpOperand *resOperand = nullptr;
        for (mlir::OpOperand &operand : op->getOpOperands()) {
            if (auto path = getPath(operand.get()); !path.isEmpty()) {
                assert(!resOperand && "Unexpected operation with multiple used paths");
                resPath = path;
                resOperand = &operand;
            }
        }
        assert(resOperand && "A path must exist for leaf operations");
        return {resOperand, resPath};
    }

    /// Walk `root`'s field-access and leaf users and replace values and operations using the
    /// `replace` functions. If `eraseDead` is true, then erase dead intermediate field access
    /// operations after replacing is done.
    mlir::LogicalResult transform(bool eraseDead = false) {
        /// Use a vector and deduplicate with a set so we get deterministic order.
        llvm::SmallVector<mlir::Operation *> leafOps;
        mlir::DenseSet<mlir::Operation *> seenLeafOps;

        auto nodeCb = [&](mlir::Value value, P4HIR::FieldPath path) {
            mlir::Operation *def = value.getDefiningOp();
            if (eraseDead && def) opsToRemove.push_back(def);

            rewriter.setInsertionPointAfterValue(value);
            mlir::Value valReplacement = replaceValue(value, path);

            if (valReplacement) {
                // Stop early: replace `value` with a new one that doesn't depend on the root.
                rewriter.replaceAllUsesWith(value, valReplacement);
                return mlir::WalkResult::skip();
            }

            valueToPath.insert({value, path});

            return mlir::WalkResult::advance();
        };

        auto leafCb = [&](mlir::Operation *user, const mlir::OpOperand &operand,
                          P4HIR::FieldPath path) {
            // It is reasonable to expect that leaf operations may have multiple uses of
            // root fields, so we don't directly ask for their replacement. Store `user` and
            // process once we have discovered all leaf paths.
            if (seenLeafOps.insert(user).second) leafOps.push_back(user);
            return mlir::WalkResult::advance();
        };

        [[maybe_unused]] auto status = PathWalkerT(nodeCb, leafCb).walk(root, rootType);
        assert(!status.wasInterrupted() && "Unexpected state");

        for (mlir::Operation *op : leafOps) {
            rewriter.setInsertionPoint(op);
            if (failed(replaceUsesIn(op))) return mlir::failure();
        }

        if (eraseDead) {
            // `opsToRemove` is in DFS preorder, so we need to erase them from back to front.
            while (!opsToRemove.empty()) {
                mlir::Operation *op = opsToRemove.pop_back_val();
                if (op->use_empty()) rewriter.eraseOp(op);
            }
        }

        return mlir::success();
    }

 protected:
    mlir::RewriterBase &rewriter;

    /// The value whose uses are being replaced.
    mlir::Value root;
    /// The plain (without p4hir.ref) type to use for field paths of `root`.
    /// Note that although it needs to have the same shape as `root.getType()`, it may differ from
    /// it.
    mlir::Type rootType;
    /// Optional helper: A new root value to use while doing the replacement or `root`.
    mlir::Value newRoot;
    /// Optional helper: A plain (without p4hir.ref) new root type for `value` after replacement.
    mlir::Type newRootType;

    /// Maintains a mapping from values that are used in leaf operations to field paths.
    llvm::DenseMap<mlir::Value, P4HIR::FieldPath> valueToPath;

    /// Maintains field-access operations in safe-to-erase order, if needed.
    llvm::SmallVector<mlir::Operation *> opsToRemove;
};

using P4HIRPathRewriter = PathRewriter<P4HIRFieldPathOps>;

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_PATH_REWRITER_H
