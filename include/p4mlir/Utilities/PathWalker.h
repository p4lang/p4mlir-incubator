// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_PATH_WALKER_H
#define P4MLIR_IMPL_UTILS_PATH_WALKER_H

#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Utilities/IRUtils.h"

namespace P4::P4MLIR::IRUtils {

/// Extensible helper class to walk field-accesses and their uses starting from some root value.
template <class FieldPathOpsT>
class PathWalker {
 public:
    using NodeCallbackFn =
        std::function<mlir::WalkResult(mlir::Operation *, mlir::Value, P4HIR::FieldPath)>;
    using LeafCallbackFn = std::function<mlir::WalkResult(
        mlir::Operation *, const mlir::OpOperand &, P4HIR::FieldPath)>;

    ///  - `nodeCb` is the callback function for intermediate nodes. This callback is called for
    ///  field-access/field-projection operations. An operation, a result value and a corresponding
    ///  field path are provided as arguments to the callback. The operation will be empty for block
    ///  arguments and the result value may be empty for symbol-based ops.
    ///  - `leafCb` is the callback function for leaf operations. This callback is called for
    ///  operations that are not intermediate nodes and that access a value with a known path.
    ///  The operation, the operand and the corresponding path are provided as arguments to the
    ///  callback. If a leaf operation has multiple operands with known paths then the callback will
    ///  be called multiple times.
    PathWalker(NodeCallbackFn nodeCb, LeafCallbackFn leafCb) : nodeCb(nodeCb), leafCb(leafCb) {}

    /// Create a `PathWalker` with a node callback.
    static PathWalker forNodes(NodeCallbackFn nodeCb) { return PathWalker(nodeCb, {}); }

    /// Create a `PathWalker` with a leaf callback.
    static PathWalker forLeafs(LeafCallbackFn leafCb) { return PathWalker({}, leafCb); }

    /// Walk field-access operations and uses starting from `root`. If `rootType` is additionally
    /// provided it is used to compute field paths instead of `root.getType()`. If `rootType` is
    /// provided it must have the same shape as `root.getType()`.
    mlir::WalkResult walk(mlir::Value root, mlir::Type rootType = mlir::Type()) {
        if (!rootType) rootType = root.getType();
        P4HIR::FieldPath rootPath(P4HIR::maybeUnref(rootType));
        return walkImpl(root, rootPath);
    }

 protected:
    // Value can only be `root` or otherwise a result of a field access operation.
    mlir::WalkResult walkImpl(mlir::Value value, P4HIR::FieldPath path) {
        auto status =
            nodeCb ? nodeCb(value.getDefiningOp(), value, path) : mlir::WalkResult::advance();
        if (status.wasSkipped()) return mlir::WalkResult::advance();
        if (status.wasInterrupted()) return mlir::WalkResult::interrupt();

        for (auto &use : value.getUses()) {
            mlir::Operation *user = use.getOwner();

            if (auto controlLocalOp = mlir::dyn_cast<P4HIR::ControlLocalOp>(user)) {
                // Special handling for additional uses made through symbol references.
                auto status = nodeCb ? nodeCb(controlLocalOp, mlir::Value(), path)
                                     : mlir::WalkResult::advance();
                if (status.wasSkipped()) continue;
                if (status.wasInterrupted()) return mlir::WalkResult::interrupt();

                auto parent = controlLocalOp->getParentOfType<P4HIR::ControlOp>();
                auto symbolUses =
                    mlir::SymbolTable::getSymbolUses(controlLocalOp.getNameAttr(), parent);
                for (auto symbolUse : *symbolUses) {
                    auto symbolRefOp = mlir::cast<P4HIR::SymToValueOp>(symbolUse.getUser());
                    if (walkImpl(symbolRefOp.getResult(), path).wasInterrupted())
                        return mlir::WalkResult::interrupt();
                }
            } else {
                auto [resValue, resPath] = FieldPathOpsT::getAccessedField(user, use, path);
                if (resValue) {
                    if (walkImpl(resValue, resPath).wasInterrupted())
                        return mlir::WalkResult::interrupt();
                } else {
                    auto status = leafCb ? leafCb(user, use, path) : mlir::WalkResult::advance();
                    if (status.wasInterrupted()) return mlir::WalkResult::interrupt();
                }
            }
        }

        return mlir::WalkResult::advance();
    }

    NodeCallbackFn nodeCb;
    LeafCallbackFn leafCb;
};

using P4HIRPathWalker = PathWalker<P4HIRFieldPathOps>;

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_PATH_WALKER_H
