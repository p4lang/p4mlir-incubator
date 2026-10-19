// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H
#define P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H

#include "p4mlir/Utilities/IRUtils.h"
#include "p4mlir/Utilities/PathRewriter.h"
#include "p4mlir/Utilities/PathWalker.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to store remapped "locations" between a source and a target collections of types.
struct LocationMapping {
    /// Represents a FieldPath in the I-th object of a collection.
    using FieldLocation = std::pair<unsigned, P4HIR::FieldPath>;

    /// Represents a FieldPath in the I-th object of a collection, where the path is split in two
    /// parts.
    using SplitFieldLocation = std::tuple<unsigned, P4HIR::FieldPath, P4HIR::FieldPath>;

    /// Constructor for when both the source and target collection types are known in advance.
    LocationMapping(llvm::SmallVector<mlir::Type> from, llvm::SmallVector<mlir::Type> to)
        : fromTypes(std::move(from)), toTypes(std::move(to)) {}

    /// Constructor for when the target collection types will be computed later.
    LocationMapping(llvm::SmallVector<mlir::Type> from, unsigned toCount)
        : fromTypes(std::move(from)), toTypes(toCount, mlir::Type()) {}

    /// Set or change the target collection types to `to`.
    void setToTypes(llvm::SmallVector<mlir::Type> to) {
        assert((to.size() == toCount()) && "Unexpected 'to' collection size");
        toTypes = std::move(to);
    }

    /// Set or change the target collection type at `index` to `to`.
    void setToType(unsigned index, mlir::Type to) { toTypes[index] = to; }

    /// Get the types of the source collection.
    const llvm::SmallVector<mlir::Type> &getFromTypes() const { return fromTypes; }

    /// Get the types of the target collection.
    const llvm::SmallVector<mlir::Type> &getToTypes() const { return toTypes; }

    /// Get the size of the source collection.
    unsigned fromCount() const { return fromTypes.size(); }

    /// Get the size of the target collection.
    unsigned toCount() const { return toTypes.size(); }

    /// Return a mapping entry if it exists for `from`, nullptr otherwise.
    const FieldLocation *lookup(FieldLocation from) const {
        auto it = mapping.find(from);
        return (it != mapping.end()) ? &(it->second) : nullptr;
    }

    /// Return a mapping entry if it exists for `from`, nullptr otherwise.
    FieldLocation *lookup(FieldLocation from) {
        auto it = mapping.find(from);
        return (it != mapping.end()) ? &(it->second) : nullptr;
    }

    /// If `from` is a field from the source collection that exists as a whole in another place,
    /// return a split location to access it in the target collection. If different subfields of
    /// `from` are located in different places then a single target location does not exist and
    /// std::nullopt is returned.
    std::optional<SplitFieldLocation> remap(FieldLocation from) const;

    /// Record that `from` in the source collection should be remapped to `to` in the target
    /// collection.
    bool addMapping(FieldLocation from, FieldLocation to);

    /// Record that the locations defined by `fromIndex` and `fromPaths` in the source collection
    /// should be remapped to the locations defined by `toIndex` and `toPaths` in the target
    /// collection.
    void addMapping(unsigned fromIndex, llvm::SmallVectorImpl<P4HIR::FieldPath> &fromPaths,
                    unsigned toIndex, llvm::SmallVectorImpl<P4HIR::FieldPath> &toPaths) {
        assert((fromPaths.size() == toPaths.size()) && "Expected same-size path collections");
        for (auto [fromPath, toPath] : llvm::zip_equal(fromPaths, toPaths))
            addMapping(FieldLocation(fromIndex, fromPath), FieldLocation(toIndex, toPath));
    }

    /// Record that the locations defined by `fromIndex` and `fromPaths` in the source collection
    /// should be remapped to the locations defined by `toIndex` and `toPath.children()` in the
    /// target collection.
    void addMapping(unsigned fromIndex, llvm::SmallVectorImpl<P4HIR::FieldPath> &fromPaths,
                    unsigned toIndex, P4HIR::FieldPath toPath) {
        auto toPaths = llvm::to_vector(toPath.children());
        addMapping(fromIndex, fromPaths, toIndex, toPaths);
    }

    /// Record that `from` in the source collection should be remapped to the `to`-ith target
    /// collection without yet specifying the exact mapping.
    bool markForMapping(FieldLocation from, unsigned to) {
        return addMapping(from, FieldLocation(to, P4HIR::FieldPath()));
    }

    /// Get a range for all paths in the `index`-th object in the source collection.
    auto getFromPaths(unsigned index) const {
        auto filteredKeys = llvm::make_filter_range(
            mapping.keys(), [index](const auto &k) { return k.first == index; });
        return llvm::map_range(filteredKeys, [](const auto &k) { return k.second; });
    }

    /// Get a range for all mapping entries.
    auto getMapping() const { return llvm::make_range(mapping.begin(), mapping.end()); }

    /// Get a range for all mapping entries regarding the `index`-th object in the source
    /// collection.
    auto getMapping(unsigned index) const {
        return llvm::make_filter_range(
            getMapping(), [index](const auto &e) { return e.getFirst().first == index; });
    }

    /// Returns a minimal vector of all FieldLocations that need to be mapped to create a complete
    /// mapping.
    llvm::SmallVector<FieldLocation> getUnmapped() const {
        llvm::SmallVector<FieldLocation> result;
        for (auto [index, type] : llvm::enumerate(fromTypes))
            getUnmappedImpl(index, P4HIR::FieldPath(type), result);

        return result;
    }

    /// Returns a minimal vector of all FieldLocations that need to be mapped to create a complete
    /// mapping of the `index`-th source object.
    llvm::SmallVector<FieldLocation> getUnmapped(unsigned index) const {
        llvm::SmallVector<FieldLocation> result;
        getUnmappedImpl(index, P4HIR::FieldPath(fromTypes[index]), result);
        return result;
    }

    /// Return true iff this mapping is complete. A complete mapping is one where all source leaf
    /// fields can be remapped in the target location.
    bool isComplete() const { return getUnmapped().empty(); }

    /// Return true iff this mapping is unique. A unique mapping is one where each source field has
    /// only one target location to be remapped at.
    bool isUnique() const {
        for (auto [index, type] : llvm::enumerate(fromTypes))
            if (!isUniqueImpl(index, P4HIR::FieldPath(type), false)) return false;

        return true;
    }

    /// Check that the source and target types and paths are valid.
    bool verify() const;

 protected:
    void getUnmappedImpl(unsigned index, P4HIR::FieldPath path,
                         llvm::SmallVector<FieldLocation> &result) const;

    bool isUniqueImpl(unsigned index, P4HIR::FieldPath path, bool hasLocation) const;

    /// Represents a location mapping between the source and the target collections.
    llvm::DenseMap<FieldLocation, FieldLocation> mapping;

    llvm::SmallVector<mlir::Type> fromTypes;
    llvm::SmallVector<mlir::Type> toTypes;
};

/// A helper class to replace uses of a source collection to ones in a target collections of values
/// using a corresponding `LocationMapping`.
template <class FieldPathOpsT>
struct LocationRewriter {
    using FieldLocation = LocationMapping::FieldLocation;

    LocationRewriter(mlir::RewriterBase &rewriter, LocationMapping &mapping,
                     llvm::SmallVector<mlir::Value> from, llvm::SmallVector<mlir::Value> to)
        : rewriter(rewriter),
          mapping(mapping),
          fromValues(std::move(from)),
          toValues(std::move(to)) {
        assert((fromValues.size() == mapping.fromCount()) &&
               "Value and type size mismatch for the source collection");
        assert((toValues.size() == mapping.toCount()) &&
               "Value and type size mismatch for the target collection");
        for (auto [value, type] : llvm::zip_equal(fromValues, mapping.getFromTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the source collection");
        for (auto [value, type] : llvm::zip_equal(toValues, mapping.getToTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the target collection");
    }

    virtual ~LocationRewriter() {}

    /// Return a remapped reference for `from` if possible, otherwise return an empty value.
    mlir::Value getRef(mlir::Location loc, FieldLocation from) {
        auto to = mapping.remap(from);
        if (!to) return mlir::Value();

        auto [toIndex, toPath, toField] = *to;
        if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return mlir::Value();

        auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
        return FieldPathOpsT::buildPathAccess(rewriter, loc, toValues[toIndex], fullPath);
    }

    /// Return a remapped value for `from` if possible, otherwise return an empty value.
    /// If `allowSplit` is true then multiple values from different places in the target
    /// collection may be combined to do so.
    mlir::Value getVal(mlir::Location loc, FieldLocation from, bool allowSplit = true) {
        if (auto to = mapping.remap(from)) {
            auto [toIndex, toPath, toField] = *to;
            auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
            mlir::Value res =
                FieldPathOpsT::buildPathAccess(rewriter, loc, toValues[toIndex], fullPath);
            if (mlir::isa<P4HIR::ReferenceType>(res.getType()))
                res = P4HIR::ReadOp::create(rewriter, loc, res);
            return res;
        } else {
            if (!allowSplit) return mlir::Value();

            unsigned index = from.first;
            auto path = from.second;

            // Build the read value by combining individual values for its fields.
            if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
                auto fields = llvm::map_to_vector(path.children(), [&](auto childPath) {
                    return getVal(loc, FieldLocation(index, childPath), allowSplit);
                });
                return P4HIR::StructOp::create(rewriter, loc, structType, fields);
            }

            return mlir::Value();
        }
    }

    /// Perform a remapped assignment for `from` if possible and return true, otherwise return
    /// false. If `allowSplit` is true then the assignment may be broken into multiple assignments
    /// in the target collection to do so.
    bool setVal(mlir::Location loc, FieldLocation from, mlir::Value val, bool allowSplit = true) {
        if (auto to = mapping.remap(from)) {
            auto [toIndex, toPath, toField] = *to;
            if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return false;

            auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
            mlir::Value res =
                FieldPathOpsT::buildPathAccess(rewriter, loc, toValues[toIndex], fullPath);
            P4HIR::AssignOp::create(rewriter, loc, val, res);
            return true;
        } else {
            if (!allowSplit) return false;

            auto [index, path] = from;

            // Break down the assignment to the individual parts.
            if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
                for (auto field : structType.getFields()) {
                    mlir::Value fieldVal =
                        P4HIR::StructExtractOp::create(rewriter, loc, val, field);
                    setVal(loc, FieldLocation(index, path[field]), fieldVal, allowSplit);
                }
                return true;
            }

            return false;
        }
    }

    /// Replace all uses of the source collection with ones from the target collection.
    mlir::LogicalResult transform() {
        for (unsigned index = 0, e = fromValues.size(); index < e; index++)
            if (failed(ValueRewriter(*this, index).transform(true))) return mlir::failure();

        return mlir::success();
    }

 protected:
    using PathRewriterT = PathRewriter<FieldPathOpsT>;
    struct ValueRewriter : public PathRewriterT {
        ValueRewriter(LocationRewriter &parent, unsigned index)
            : PathRewriterT(parent.rewriter, parent.fromValues[index]),
              parent(parent),
              index(index) {}

        virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) override {
            if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
                auto path = this->getOperandWithPath(op).second;
                auto newVal = parent.getVal(op->getLoc(), FieldLocation(index, path));
                if (!newVal) return op->emitError("Cannot replace read of ") << path.str();
                this->rewriter.replaceOp(op, newVal);
                return mlir::success();
            } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
                auto path = this->getOperandWithPath(op).second;
                bool success =
                    parent.setVal(op->getLoc(), FieldLocation(index, path), assignOp.getValue());
                if (!success) return op->emitError("Cannot replace write to ") << path.str();
                this->rewriter.eraseOp(op);
                return mlir::success();
            } else if (mlir::isa<P4HIR::ControlLocalOp>(op)) {
                this->rewriter.eraseOp(op);
                return mlir::success();
            } else {
                bool success = true;
                this->rewriter.modifyOpInPlace(op, [&]() {
                    for (mlir::OpOperand &operand : op->getOpOperands()) {
                        auto val = operand.get();
                        if (auto path = this->getPath(val); !path.isEmpty()) {
                            FieldLocation from(index, path);
                            bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
                            auto newVal = isRef ? parent.getRef(val.getLoc(), from)
                                                : parent.getVal(val.getLoc(), from);
                            if (!newVal) {
                                op->emitError("Cannot replace access of ") << path.str();
                                success = false;
                                break;
                            }
                            operand.set(newVal);
                        }
                    }
                });
                return success ? mlir::success() : mlir::failure();
            }
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
