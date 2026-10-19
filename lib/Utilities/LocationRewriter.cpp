// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Utilities/LocationRewriter.h"

#include "p4mlir/Utilities/IRUtils.h"

using namespace P4::P4MLIR;

std::optional<IRUtils::LocationMapping::SplitFieldLocation> IRUtils::LocationMapping::remap(
    FieldLocation from) const {
    assert((from.first < fromCount()) && "'from' index out of bounds");
    auto [fromIndex, fromPath] = from;
    auto prefixes = llvm::to_vector(fromPath.paths());

    while (!prefixes.empty()) {
        auto fromPrefix = prefixes.pop_back_val();
        if (auto *toLocation = lookup(FieldLocation(fromIndex, fromPrefix))) {
            auto fromSuffix = P4HIR::FieldPath::subtract(fromPath, fromPrefix);
            auto [toIndex, toPath] = *toLocation;
            return SplitFieldLocation(toIndex, toPath, fromSuffix);
        }
    }

    return std::nullopt;
}

bool IRUtils::LocationMapping::addMapping(FieldLocation from, FieldLocation to) {
    assert((from.first < fromCount()) && "'from' index out of bounds");
    assert((to.first < toCount()) && "'to' index out of bounds");
    assert((from.second.getRootType() == fromTypes[from.first]) && "'from' type mismatch");
    assert((!to.second || to.second.getRootType() == toTypes[to.first]) && "'to' type mismatch");
    auto [it, ins] = mapping.insert({from, FieldLocation(0, P4HIR::FieldPath())});
    it->second = to;
    return ins;
}

bool IRUtils::LocationMapping::verify() const {
    for (auto fromType : fromTypes)
        if (!fromType) return false;

    for (auto toType : toTypes)
        if (!toType) return false;

    for (auto [from, to] : mapping) {
        auto [fromIndex, fromPath] = from;
        auto [toIndex, toPath] = to;
        if (fromPath.getRootType() != fromTypes[fromIndex]) return false;
        if (toPath.getRootType() != toTypes[toIndex]) return false;
    }

    return true;
}

void IRUtils::LocationMapping::getUnmappedImpl(unsigned index, P4HIR::FieldPath path,
                                               llvm::SmallVector<FieldLocation> &result) const {
    bool hasMapping = lookup(FieldLocation(index, path));
    if (hasMapping) return;

    auto origSize = result.size();
    bool merge = true;
    for (auto childPath : path.children()) {
        auto lastSize = result.size();
        getUnmappedImpl(index, childPath, result);

        auto newCount = result.size() - lastSize;
        if (newCount != 1 || result.back().second != childPath) merge = false;
    }

    // If there's no mapped field within this path's children, then instead of adding all
    // children just add this path to minimize the output. Also handles the no-children case.
    if (merge) {
        result.resize(origSize);
        result.emplace_back(index, path);
    }
}

bool IRUtils::LocationMapping::isUniqueImpl(unsigned index, P4HIR::FieldPath path,
                                            bool hasLocation) const {
    bool hasMapping = lookup(FieldLocation(index, path));
    if (hasMapping && hasLocation) return false;

    for (auto childPath : path.children())
        if (!isUniqueImpl(index, childPath, hasMapping || hasLocation)) return false;

    return true;
}
