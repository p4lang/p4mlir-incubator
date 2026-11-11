// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H
#define P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H

#include <deque>

#include "p4mlir/Utilities/IRUtils.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to store remapped "locations" between a source and a target collections of types.
struct LocationMapping {
    struct Source;
    struct Target;

    /// Represents a location. A location is a field in one of the source or target collections.
    /// A location may be empty or have an empty path. A location with an empty path belongs to a
    /// collection and represents a field in that location which is still unspecified.
    struct FieldLocation {
        // Special constructor for an empty FieldLocation.
        FieldLocation() : FieldLocation(P4HIR::FieldPath(), std::numeric_limits<unsigned>::max()) {}

        FieldLocation(P4HIR::FieldPath path, unsigned index) : path(path), index(index) {}

        /// Return true if this is a non-empty location.
        operator bool() const { return index != std::numeric_limits<unsigned>::max(); }

        /// Get the path in the collection.
        P4HIR::FieldPath getPath() const { return path; }

        /// Return if a path is present.
        bool hasPath() const { return !path.isEmpty(); }

        /// Return the index within the collection.
        unsigned getIndex() const { return index; }

     private:
        P4HIR::FieldPath path;
        unsigned index;
    };

    /// A struct holding information for a partial mapping for a source to a target path.
    /// Represents that (sourceRoot ++ sourceField) has a mapping to (targetRoot ++ targetField).
    struct FieldMapping {
        /// Return the location for the source field.
        FieldLocation getSource() {
            auto fullPath = P4HIR::FieldPath::concat(sourceRoot.getPath(), sourceField);
            return FieldLocation(fullPath, sourceRoot.getIndex());
        }

        /// Return the location for the target field.
        FieldLocation getTarget() {
            auto fullPath = P4HIR::FieldPath::concat(targetRoot.getPath(), targetField);
            return FieldLocation(fullPath, targetRoot.getIndex());
        }

        FieldLocation sourceRoot;
        P4HIR::FieldPath sourceField;
        FieldLocation targetRoot;
        P4HIR::FieldPath targetField;
    };

 private:
    /// A node to build a trie-like data structure for mapping fields to `FieldLocation`s.
    /// Fields are implicitly represented as trails of field indices, starting from any type.
    /// Invariant: If `!children.empty()` then all its leaf nodes have a non-empty mapping.
    /// Only the root may be a leaf node but have no mapping.
    struct MappingNode {
        /// Get or create a child node representing the `index`-th field.
        MappingNode *getOrCreate(unsigned index) {
            auto [it, ins] = children.try_emplace(index);  // value-initializes an empty unique_ptr
            if (ins) it->second = std::make_unique<MappingNode>();
            return it->second.get();
        }

        /// Get a child node representing the `index`-th field if present, return nullptr otherwise.
        MappingNode *get(unsigned index) {
            auto it = children.find(index);
            return (it != children.end()) ? it->second.get() : nullptr;
        }

        /// Returns true if there are no mappings for subfields of the current node.
        bool isLeaf() const { return children.empty(); }

        /// Return children as a range of tuple<index, node*>.
        auto getEntries() {
            return llvm::map_range(children,
                                   [](auto &e) { return std::pair{e.first, e.second.get()}; });
        }

        /// Map from field index to a nested node.
        llvm::DenseMap<unsigned, std::unique_ptr<MappingNode>> children;

        /// The mapping payload.
        FieldLocation to;
    };

 public:
    /// Represents an entry in the target collection.
    /// Target collections may initially have no root type, in which case can only create unknown
    /// locations.
    struct Target {
        /// Set the root type after creation. Can only be called once with a non-empty type.
        void setRootType(mlir::Type type) {
            assert(type && "Expected a non-empty type");
            assert(!root && "Cannot change root type once set");
            root = type;
        }

        /// Return a location for `path` within this entry. The entry must have a type specified and
        /// the root type of `path` must match with it.
        FieldLocation loc(P4HIR::FieldPath path) {
            assert(root && "Cannot get location when root type is not set");
            assert((path.getRootType() == root) && "Path root type does not match");
            return FieldLocation(path, index);
        }

        /// Return a range for all locations in `paths`.
        template <typename CollectionT>
        auto locs(CollectionT paths) {
            return llvm::map_range(paths, [this](P4HIR::FieldPath path) { return loc(path); });
        }

        /// Return an unknown location within this entry.
        FieldLocation loc() { return FieldLocation(P4HIR::FieldPath(), index); }

     private:
        friend struct LocationMapping;

        Target(mlir::Type root, unsigned index) : root(root), index(index) {}

        mlir::Type root;
        unsigned index;
    };

    /// Represents an entry in the source collection.
    struct Source {
        /// Return a location for `path` within this entry. The root type of `path` must match the
        /// entry's type.
        FieldLocation loc(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            return FieldLocation(path, index);
        }

        /// Return a range for all locations in `paths`.
        template <typename CollectionT>
        auto locs(CollectionT paths) {
            return llvm::map_range(paths, [this](P4HIR::FieldPath path) { return loc(path); });
        }

        /// Add a mapping specifying that `from` must be mapped to `to`.
        /// Return true if there was a previous different mapping for `from`.
        bool addMapping(FieldLocation from, FieldLocation to) {
            assert((from.getIndex() == index) && "FieldLocation index mismatch");
            MappingNode *node = getOrCreate(from.getPath());
            bool isNewMapping = !node->to;
            node->to = to;
            return isNewMapping;
        }

        /// Add a mapping specifying that `from` must be mapped to `to`.
        /// Return true if there was a previous different mapping for `from`.
        bool addMapping(P4HIR::FieldPath from, FieldLocation to) {
            return addMapping(loc(from), to);
        }

        /// Add mappings specifying that elements of `fromCollection` must be mapped to their
        /// corresponding ones in `toCollection`.
        template <typename Collection1T, typename Collection2T>
        void addMappings(Collection1T fromCollection, Collection2T toCollection) {
            for (auto [from, to] : llvm::zip_equal(fromCollection, toCollection))
                addMapping(from, to);
        }

        /// Get mappings that affect `from`. If `includeSubfields` is false then only a mapping for
        /// `from` may be returned.
        llvm::SmallVector<FieldMapping> remap(FieldLocation from, bool includeSubfields = true);

        /// Clear all mappings.
        void clear() { mapping = MappingNode(); }

        /// Return the mapped location for `from` if it exists, otherwise an empty location.
        /// Nested mappings for subfields of `from` are not considered.
        FieldLocation getMapping(FieldLocation from) {
            assert((from.getIndex() == index) && "FieldLocation index mismatch");
            MappingNode *node = get(from.getPath());
            return node ? node->to : FieldLocation();
        }

        /// Return true if there are mappings for subfields of `from`.
        bool hasSubfieldMappings(FieldLocation from) {
            assert((from.getIndex() == index) && "FieldLocation index mismatch");
            MappingNode *node = get(from.getPath());
            return node && !node->children.empty();
        }

        /// Return true iff there are mappings nested under mappings.
        bool hasSubfieldMappings();

        /// Get vector of paths with their mapped locations. If `toplevel` is true then do not
        /// include mappings nested under other mappings.
        llvm::SmallVector<std::pair<P4HIR::FieldPath, FieldLocation>> getMappings(
            bool onlyToplevel = false);

        /// Get vector of paths that have a mapped location. If `toplevel` is true then do not
        /// include mappings nested under other mappings.
        llvm::SmallVector<P4HIR::FieldPath> getMappedPaths(bool onlyToplevel = false);

        /// Get vector of toplevel paths that do not have a mapped location.
        llvm::SmallVector<P4HIR::FieldPath> getUnmappedPaths();

        /// Return true iff this mapping is complete. A complete mapping is one where all source
        /// fields can be remapped in some target location.
        bool isComplete() { return getUnmappedPaths().empty(); }

     private:
        friend struct LocationMapping;

        Source(mlir::Type root, unsigned index) : root(root), index(index) {
            assert(root && "Expected a non-empty type");
        }

        /// Get or create a node representing `path`.
        MappingNode *getOrCreate(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            MappingNode *cur = &mapping;
            for (auto index : path.field_indices()) cur = cur->getOrCreate(index);
            return cur;
        }

        /// Get a node representing `path` if it exists, return nullptr otherwise.
        MappingNode *get(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            MappingNode *cur = &mapping;
            for (auto index : path.field_indices()) {
                cur = cur->get(index);
                if (!cur) break;
            }
            return cur;
        }

        /// Perform a preorder DFS traversal of the trie's nodes, calling `cb` on each node.
        void visit(llvm::function_ref<bool(P4HIR::FieldPath, MappingNode *)> cb);

        /// Dump all mappings.
        void dumpMapping() {
            if (!mapping.isLeaf() || mapping.to)
                dumpMappingImpl(&mapping, P4HIR::FieldPath(root), 2);
        }

        void dumpMappingImpl(MappingNode *node, P4HIR::FieldPath path, unsigned nestLevel);

        mlir::Type root;
        MappingNode mapping;
        unsigned index;
    };

    /// Dump sources, targets and all mappings.
    void dump();

    /// Add a mapping specifying that `from` must be mapped to `to`.
    /// Return true if there was a previous different mapping for `from`.
    bool addMapping(FieldLocation from, FieldLocation to) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.addMapping(from, to);
    }

    /// Add mappings specifying that elements of `fromCollection` must be mapped to their
    /// corresponding ones in `toCollection`.
    template <typename Collection1T, typename Collection2T>
    void addMappings(Collection1T fromCollection, Collection2T toCollection) {
        for (auto [from, to] : llvm::zip_equal(fromCollection, toCollection)) addMapping(from, to);
    }

    /// Get mappings that affect `from`. If `includeSubfields` is false then only a mapping for
    /// `from` may be returned.
    llvm::SmallVector<FieldMapping> remap(FieldLocation from, bool includeSubfields = true) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.remap(from, includeSubfields);
    }

    /// Return the mapped location for `from` if it exists, otherwise an empty location.
    /// Nested mappings for subfields of `from` are not considered.
    FieldLocation getMapping(FieldLocation from) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.getMapping(from);
    }

    /// Return true if there are mappings for subfields of `from`.
    bool hasSubfieldMappings(FieldLocation from) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.hasSubfieldMappings(from);
    }

    /// Return true iff all mappings are complete. A complete mapping is one where all source
    /// fields can be remapped in some target location.
    bool isComplete() {
        for (auto &source : sources)
            if (!source.isComplete()) return false;
        return true;
    }

    /// Return true iff there are mappings nested under mappings.
    bool hasSubfieldMappings() {
        for (auto &source : sources)
            if (source.hasSubfieldMappings()) return true;
        return false;
    }

    /// Create a new entry in the sources collection.
    Source &createSource(mlir::Type root) {
        // Private constructor does not allow emplace_back.
        sources.push_back(Source(root, sourcesCount()));
        return sources.back();
    }

    /// Create a new entry in the targets collection.
    Target &createTarget(mlir::Type root = mlir::Type()) {
        // Private constructor does not allow emplace_back.
        targets.push_back(Target(root, targetsCount()));
        return targets.back();
    }

    /// Get the size of the source collection.
    unsigned sourcesCount() const { return static_cast<unsigned>(sources.size()); }

    /// Get the size of the target collection.
    unsigned targetsCount() const { return static_cast<unsigned>(targets.size()); }

    /// Get the types of the source collection.
    auto getSourceTypes() const {
        return llvm::map_range(sources, [](auto &e) { return e.root; });
    }

    /// Get the types of the target collection.
    auto getTargetTypes() const {
        return llvm::map_range(targets, [](auto &e) { return e.root; });
    }

 private:
    std::deque<Source> sources;
    std::deque<Target> targets;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H
