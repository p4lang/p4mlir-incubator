// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Utilities/LocationMapping.h"

#include "p4mlir/Utilities/IRUtils.h"

using namespace P4::P4MLIR;
using namespace P4::P4MLIR::IRUtils;

using FieldLocation = LocationMapping::FieldLocation;
using FieldMapping = LocationMapping::FieldMapping;

llvm::SmallVector<FieldMapping> LocationMapping::Source::remap(FieldLocation from,
                                                               bool includeSubfields /* = true */) {
    assert((from.getIndex() == index) && "FieldLocation index mismatch");
    llvm::SmallVector<FieldMapping> result;

    // First find the most nested mapping that maps a superset of the requested location.
    // If such a mapping exists it will be unique.
    FieldMapping rootMapping;
    MappingNode *cur = &mapping;

    if (cur->to) {
        rootMapping.targetRoot = cur->to;
        auto targetType = rootMapping.targetRoot.getPath().getType();
        rootMapping.targetField = P4HIR::FieldPath(targetType);
    }

    for (auto index : from.getPath().field_indices()) {
        // Build the target field by appending `index` in lockstep with `from` iteration.
        if (rootMapping.targetField) rootMapping.targetField.append(index);

        if (cur) {
            cur = cur->get(index);

            if (cur && cur->to) {
                // We found a more deeply nested mapping, start again from that root.
                rootMapping.targetRoot = cur->to;
                auto targetType = rootMapping.targetRoot.getPath().getType();
                rootMapping.targetField = P4HIR::FieldPath(targetType);
            }
        }
    }

    if (rootMapping.targetRoot) {
        rootMapping.sourceRoot = from;

        // Canonicalize targetField.
        if (rootMapping.targetField.getFieldID() == 0) rootMapping.targetField = P4HIR::FieldPath();

        result.push_back(rootMapping);
    }

    if (!cur || !includeSubfields) {
        // There are no other mappings to take into account.
        return result;
    }

    // Next, we need to find subfields of `from` that may be mapped to other locations.
    std::function<void(MappingNode *, P4HIR::FieldPath, bool)> findSubfieldMappings =
        [&](MappingNode *node, P4HIR::FieldPath path, bool first) {
            if (!first && node->to) {
                FieldMapping mapping;
                mapping.targetRoot = node->to;
                mapping.sourceRoot = from;
                mapping.sourceField = path;
            }

            for (auto [index, child] : node->getEntries()) {
                findSubfieldMappings(child, path[index], false);
            }
        };

    // If `cur` is non-null after the loop above, it represents the node that corresponds exactly to
    // `from`. Start from that and visit children mappings.
    findSubfieldMappings(cur, P4HIR::FieldPath(from.getPath().getType()), true);

    return result;
}

llvm::SmallVector<std::pair<P4HIR::FieldPath, FieldLocation>> LocationMapping::Source::getMappings(
    bool onlyToplevel /* = false */) {
    llvm::SmallVector<std::pair<P4HIR::FieldPath, FieldLocation>> result;
    visit([&](P4HIR::FieldPath from, MappingNode *node) {
        if (node->to) result.emplace_back(from, node->to);
        return !onlyToplevel;
    });
    llvm::sort(result, [](auto &p1, auto &p2) { return p1.first < p2.first; });
    return result;
}

llvm::SmallVector<P4HIR::FieldPath> LocationMapping::Source::getMappedPaths(
    bool onlyToplevel /* = false */) {
    llvm::SmallVector<P4HIR::FieldPath> result;
    visit([&](P4HIR::FieldPath from, MappingNode *node) {
        if (node->to) result.push_back(from);
        return !onlyToplevel;
    });
    llvm::sort(result);
    return result;
}

llvm::SmallVector<P4HIR::FieldPath> LocationMapping::Source::getUnmappedPaths() {
    llvm::SmallVector<P4HIR::FieldPath> result;
    std::function<void(MappingNode *, P4HIR::FieldPath)> getUnmappedImpl =
        [&](MappingNode *node, P4HIR::FieldPath path) {
            // Found mapping for `node`.
            if (node->to || !path.hasChildren()) return;

            auto origSize = result.size();
            bool merge = true;
            for (auto [index, childPath] : llvm::enumerate(path.children())) {
                auto lastSize = result.size();

                auto it = node->children.find(index);
                if (it == node->children.end())
                    result.push_back(childPath);
                else
                    getUnmappedImpl(it->second.get(), childPath);

                auto newCount = result.size() - lastSize;
                if (newCount != 1 || result.back() != childPath) merge = false;
            }

            // If there's no mapped field within this path's children, then instead of adding all
            // children just add this path to minimize the output. Also handles the no-children
            // case.
            if (merge) {
                result.resize(origSize);
                result.emplace_back(path);
            }
        };
    getUnmappedImpl(&mapping, P4HIR::FieldPath(root));
    llvm::sort(result);
    return result;
}

bool LocationMapping::Source::hasSubfieldMappings() {
    bool result = false;
    visit([&](P4HIR::FieldPath path, MappingNode *node) {
        if (node->to) {
            if (!node->children.empty()) {
                // Going by the invariant, if this node has children then it also has more nested
                // mappings.
                result = true;
            }
            return false;
        }

        return true;
    });
    return result;
}

/// Perform a preorder DFS traversal of the trie's nodes, calling `cb` on each node.
void LocationMapping::Source::visit(llvm::function_ref<bool(P4HIR::FieldPath, MappingNode *)> cb) {
    std::function<void(MappingNode *, P4HIR::FieldPath)> visitImpl = [&](MappingNode *node,
                                                                         P4HIR::FieldPath path) {
        if (!cb(path, node)) return;

        for (auto [index, child] : node->getEntries()) visitImpl(child, path[index]);
    };

    visitImpl(&mapping, P4HIR::FieldPath(root));
}

void LocationMapping::Source::dumpMappingImpl(MappingNode *node, P4HIR::FieldPath path,
                                              unsigned nestLevel) {
    std::string ident(nestLevel * 2, ' ');
    llvm::dbgs() << ident;
    if (path.getType() == root)
        llvm::dbgs() << "root";
    else
        llvm::dbgs() << path;

    if (node->to) {
        llvm::dbgs() << " -> #" << node->to.getIndex();
        if (node->to.getPath()) llvm::dbgs() << ", " << node->to.getPath();
    }

    llvm::dbgs() << "\n";

    if (node->isLeaf()) return;

    auto entries = llvm::to_vector(node->getEntries());
    llvm::sort(entries, [](auto p1, auto p2) { return p1.first < p2.first; });
    for (auto [index, child] : entries) dumpMappingImpl(child, path[index], nestLevel + 1);
}

void LocationMapping::dump() {
    llvm::dbgs() << "Source collections (" << sourcesCount() << "):\n";
    for (auto [index, source] : llvm::enumerate(sources))
        llvm::dbgs() << "  " << index << ": " << source.root << "\n";
    llvm::dbgs() << "Target collections (" << sourcesCount() << "):\n";
    for (auto [index, target] : llvm::enumerate(targets))
        llvm::dbgs() << "  " << index << ": " << target.root << "\n";
    llvm::dbgs() << "Mappings:\n";
    for (auto [index, source] : llvm::enumerate(sources)) {
        llvm::dbgs() << "  " << index << ":\n";
        source.dumpMapping();
    }
}
