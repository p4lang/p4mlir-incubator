// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H
#define P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H

#include "p4mlir/Utilities/IRUtils.h"
#include "p4mlir/Utilities/LocationMapping.h"
#include "p4mlir/Utilities/PathRewriter.h"
#include "p4mlir/Utilities/PathWalker.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to replace uses of a source collection to ones in a target collections of values
/// using a corresponding `LocationMapping`.
template <class FieldPathOpsT>
struct LocationRewriter {
    using FieldPathOps = FieldPathOpsT;
    using FieldLocation = LocationMapping::FieldLocation;
    using FieldMapping = LocationMapping::FieldMapping;
    using OperandWithLocation = std::pair<mlir::OpOperand *, FieldLocation>;

    /// The sizes of `from` and `to` must match the `source` and `target` collection sizes in
    /// `mapping`. `from` values must be non-empty. `to` values may be empty if unknown field
    /// mappings exist.
    LocationRewriter(mlir::RewriterBase &rewriter, LocationMapping &mapping,
                     llvm::SmallVector<mlir::Value> from, llvm::SmallVector<mlir::Value> to)
        : rewriter(rewriter),
          mapping(mapping),
          fromValues(std::move(from)),
          toValues(std::move(to)) {
        assert((fromValues.size() == mapping.sourcesCount()) &&
               "Value and type size mismatch for the source collection");
        assert((toValues.size() == mapping.targetsCount()) &&
               "Value and type size mismatch for the target collection");
        for (auto [value, type] : llvm::zip_equal(fromValues, mapping.getSourceTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the source collection");
        for (auto [value, type] : llvm::zip_equal(toValues, mapping.getTargetTypes()))
            if (value)
                assert((P4HIR::maybeUnref(value.getType()) == type) &&
                       "Value and type mismatch in the target collection");
    }

    virtual ~LocationRewriter() {}

 protected:
    /// Emit code for operation `where` to obtain a reference for the original field
    /// `originalLocation` that has been mapped to `newLocation`. Returns an empty value if such a
    /// reference cannot be obtained.
    virtual mlir::Value getTargetRefImpl(mlir::Operation *where, mlir::Location loc,
                                         FieldLocation originalLocation,
                                         FieldLocation newLocation) {
        if (!newLocation.getPath() || !toValues[newLocation.getIndex()]) {
            // Default implementation does not support unknown paths.
            return mlir::Value();
        }

        if (!mlir::isa<P4HIR::ReferenceType>(toValues[newLocation.getIndex()].getType()))
            return mlir::Value();

        return PathRewriterT::buildPathAccess(
            rewriter, where, loc, toValues[newLocation.getIndex()], newLocation.getPath());
    }

    /// Emit code for operation `where` to read the value for the original field `originalLocation`
    /// that has been mapped to `newLocation`. Returns an empty value if that field cannot be read.
    virtual mlir::Value readTargetImpl(mlir::Operation *where, mlir::Location loc,
                                       FieldLocation originalLocation, FieldLocation newLocation) {
        if (!newLocation.getPath() || !toValues[newLocation.getIndex()]) {
            // Default implementation does not support unknown paths.
            return mlir::Value();
        }

        mlir::Value res = PathRewriterT::buildPathAccess(
            rewriter, where, loc, toValues[newLocation.getIndex()], newLocation.getPath());
        if (mlir::isa<P4HIR::ReferenceType>(res.getType()))
            res = P4HIR::ReadOp::create(rewriter, loc, res);

        return res;
    }

    /// Emit code for operation `where` to assign `value` to to the original field
    /// `originalLocation` that has been mapped to `newLocation`. Return true iff the assignment
    /// generation is successful.
    virtual bool assignTargetImpl(mlir::Operation *where, mlir::Location loc, mlir::Value value,
                                  FieldLocation originalLocation, FieldLocation newLocation) {
        if (auto ref = getTargetRefImpl(where, loc, originalLocation, newLocation)) {
            P4HIR::AssignOp::create(rewriter, loc, value, ref);
            return true;
        }

        return false;
    }

 public:
    /// Return a remapped reference to access `from` at `where` if possible, otherwise return an
    /// empty value.
    mlir::Value getTargetRef(mlir::Operation *where, mlir::Location loc, FieldLocation from) {
        auto mappings = mapping.remap(from, true);
        if (mappings.size() != 1) return mlir::Value();

        // Check if the mapping is for the entirety of `from`.
        auto to = mappings[0];
        if (to.getSource().getPath() != from.getPath()) return mlir::Value();

        return getTargetRefImpl(where, loc, from, to.getTarget());
    }

    /// Return a remapped value to access `from` at `where` if possible, otherwise return an empty
    /// value. If `allowSplit` is true then multiple values from different places in the target
    /// collection may be combined to do so.
    mlir::Value readTarget(mlir::Operation *where, mlir::Location loc, FieldLocation from,
                           bool allowSplit = true) {
        // Helper function to build and return a value for `fieldLocation`.
        // If a subfield mapping exists at some point, we need to deconstruct an existing value for
        // an outer field, replace the subfield and then rebuild the result. The incoming outer
        // value in this case is provided in `fieldValue`.
        std::function<mlir::Value(mlir::Value, FieldLocation, bool)> buildValue =
            [&](mlir::Value fieldValue, FieldLocation fieldLocation, bool first) -> mlir::Value {
            // If a more-nested mapping exists then it overwrites `value`.
            auto to = mapping.getMapping(fieldLocation);
            if (!first && to) {
                fieldValue = readTargetImpl(where, loc, fieldLocation, to);
                if (!fieldValue) return mlir::Value();
            }

            // There are no subfield mappings for `fieldLocation`, return `value`.
            if (!mapping.hasSubfieldMappings(fieldLocation)) return fieldValue;

            if (!allowSplit) return mlir::Value();

            // There are subfield mappings for `field`. We need to build the resulting value
            // piece-by-piece.
            auto fieldPath = fieldLocation.getPath();
            auto structLikeType =
                mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(fieldPath.getType());
            if (!structLikeType) return mlir::Value();

            llvm::SmallVector<mlir::Value> subfieldValues;
            for (auto subfield : structLikeType.getFields()) {
                mlir::Value subfieldValue;
                if (fieldValue)
                    subfieldValue =
                        P4HIR::StructExtractOp::create(rewriter, loc, fieldValue, subfield);

                FieldLocation subfieldLocation(fieldPath[subfield], fieldLocation.getIndex());
                subfieldValue = buildValue(subfieldValue, subfieldLocation, false);
                if (!subfieldValue) return mlir::Value();

                subfieldValues.push_back(subfieldValue);
            }

            return P4HIR::StructOp::create(rewriter, loc, structLikeType, subfieldValues);
        };

        mlir::Value value;

        // First obtain a top-level mapping, if it exists.
        auto rootMapping = mapping.remap(from, false);
        if (!rootMapping.empty()) {
            value = readTargetImpl(where, loc, from, rootMapping[0].getTarget());
            if (!value) return mlir::Value();
        }

        return buildValue(value, from, true);
    }

    /// Perform a remapped assignment to `from` at `where` if possible and return true, otherwise
    /// return false. If `allowSplit` is true then the assignment may be broken into multiple
    /// assignments in the target collection to do so.
    bool assignTarget(mlir::Operation *where, mlir::Location loc, FieldLocation from,
                      mlir::Value value, bool allowSplit = true) {
        // Helper function to assign `fieldValue` to `fieldLocation`.
        // If a subfield mapping exists at some point, we need to deconstruct the value and assign
        // individual pieces to inner remapped fields.
        std::function<bool(mlir::Value, FieldLocation, bool)> assignValue =
            [&](mlir::Value fieldValue, FieldLocation fieldLocation, bool first) -> bool {
            // If a more-nested mapping exists then it overwrites the field.
            auto to = mapping.getMapping(fieldLocation);
            if (!first && to) {
                bool status = assignTargetImpl(where, loc, fieldValue, fieldLocation, to);
                if (!status) return false;
            }

            // There are no subfield mappings for `fieldLocation`, return.
            if (!mapping.hasSubfieldMappings(fieldLocation)) return true;

            if (!allowSplit) return false;

            // There are subfield mappings for `field`. We need to overwrite values piece-by-piece.
            auto fieldPath = fieldLocation.getPath();
            auto structLikeType =
                mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(fieldPath.getType());
            if (!structLikeType) return false;

            llvm::SmallVector<mlir::Value> subfieldValues;
            for (auto subfield : structLikeType.getFields()) {
                mlir::Value subfieldValue =
                    P4HIR::StructExtractOp::create(rewriter, loc, fieldValue, subfield);
                FieldLocation subfieldLocation(fieldPath[subfield], fieldLocation.getIndex());
                bool status = assignValue(subfieldValue, subfieldLocation, false);
                if (!status) return false;
            }

            return true;
        };

        // First assign to a top-level mapping, if it exists.
        auto rootMapping = mapping.remap(from, false);
        if (!rootMapping.empty()) {
            bool status = assignTargetImpl(where, loc, value, from, rootMapping[0].getTarget());
            if (!status) return false;
        }

        return assignValue(value, from, true);
    }

    /// `op` is an operation that has at least one operand accessing fields of a source collection
    /// object. These operands and the accessed locations are provided in `operands`. If it is
    /// desirable to transform `op` then this should be done here. Return mlir::success() to
    /// continue the rewriting or mlir::failure() to interrupt it.
    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op,
                                              llvm::SmallVector<OperandWithLocation> operands) {
        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
            assert((operands.size() == 1) && "Expected single argument replacement");
            auto [operand, from] = operands.front();
            auto newVal = readTarget(op, op->getLoc(), from);
            if (!newVal) return op->emitError("Cannot replace read of ") << from.getPath();
            this->rewriter.replaceOp(op, newVal);
            return mlir::success();
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            assert((operands.size() == 1) && "Expected single argument replacement");
            auto [operand, from] = operands.front();
            bool success = assignTarget(op, op->getLoc(), from, assignOp.getValue());
            if (!success) return op->emitError("Cannot replace write to ") << from.getPath();
            this->rewriter.eraseOp(op);
            return mlir::success();
        } else {
            bool success = true;
            this->rewriter.modifyOpInPlace(op, [&]() {
                for (auto [operand, from] : operands) {
                    auto val = operand->get();
                    bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
                    auto newVal = isRef ? getTargetRef(op, val.getLoc(), from)
                                        : readTarget(op, val.getLoc(), from);

                    // If we cannot get a direct reference for replacing, we can introduce a
                    // temporary variable, initialize it with getVal and then write back the result
                    // with setVal. This may succeed in more cases since getVal/setVal can handle
                    // multiple locations.
                    if (isRef && !newVal) {
                        mlir::OpBuilder::InsertionGuard guard(rewriter);

                        mlir::Value initVal = readTarget(op, val.getLoc(), from);
                        if (!initVal) {
                            op->emitError("Cannot replace access of ") << from.getPath();
                            success = false;
                            break;
                        }

                        rewriter.setInsertionPoint(op);
                        auto tempVariable = P4HIR::VariableOp::create(
                            rewriter, val.getLoc(), val.getType(), "$temp." + from.getPath().str());
                        newVal = tempVariable.getRef();
                        P4HIR::AssignOp::create(rewriter, val.getLoc(), initVal, tempVariable);

                        rewriter.setInsertionPointAfter(op);
                        auto tempVal = P4HIR::ReadOp::create(rewriter, val.getLoc(), tempVariable);
                        bool success = assignTarget(op, val.getLoc(), from, tempVal);
                        if (!success) {
                            op->emitError("Cannot replace access of ") << from.getPath();
                            success = false;
                            break;
                        }
                    }

                    if (!newVal) {
                        op->emitError("Cannot replace access of ") << from.getPath();
                        success = false;
                        break;
                    }

                    operand->set(newVal);
                }
            });
            return success ? mlir::success() : mlir::failure();
        }
    }

    /// Replace all uses of the source collection with ones from the target collection.
    mlir::LogicalResult transform() {
        for (unsigned index = 0, e = fromValues.size(); index < e; index++)
            if (failed(ValueRewriter(*this, index).transform(true))) return mlir::failure();

        return mlir::success();
    }

 protected:
    using PathRewriterT = PathRewriter<FieldPathOps>;

    struct ValueRewriter : public PathRewriterT {
        using OperandWithPathT = typename PathRewriterT::OperandWithPath;

        ValueRewriter(LocationRewriter &parent, unsigned index)
            : PathRewriterT(parent.rewriter, parent.fromValues[index]),
              parent(parent),
              index(index) {}

        virtual mlir::LogicalResult replaceUsesIn(
            mlir::Operation *op, llvm::SmallVector<OperandWithPathT> operands) override {
            auto operandsWithLocations = llvm::map_to_vector(operands, [&](auto p) {
                auto [operand, path] = p;
                return OperandWithLocation(operand, FieldLocation(path, index));
            });
            return parent.replaceUsesIn(op, operandsWithLocations);
        }

        LocationRewriter &parent;
        unsigned index;
    };

    mlir::RewriterBase &rewriter;
    LocationMapping &mapping;
    llvm::SmallVector<mlir::Value> fromValues;
    llvm::SmallVector<mlir::Value> toValues;
};

using P4HIRLocationRewriter = LocationRewriter<P4HIRFieldPathOps>;

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H
