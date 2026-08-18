// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Dialect/P4HIR/FieldPath.h"

#include "llvm/ADT/SmallString.h"

using namespace P4::P4MLIR;

using P4HIR::FieldPath;

unsigned FieldPath::getMaxFieldID(mlir::Type type) {
    auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(type);
    if (!itype) return 0;

    unsigned maxFieldID = 0;
    for (auto field : itype.getFields()) maxFieldID += 1 + getMaxFieldID(field.getType());

    return maxFieldID;
}

unsigned FieldPath::getFieldID(P4HIR::IndexableTypeInterface itype, unsigned index) {
    assert((index < itype.getFieldCount()) && "Invalid field index");
    unsigned fid = 0;
    unsigned i = 0;
    unsigned e = itype.getFieldCount();

    for (; i < e; i++) {
        // Move `fid` to the fieldID of the current field.
        fid += 1;

        if (i == index) return fid;

        // Increment the field ID for the next field by the number of subfields.
        fid += getMaxFieldID(itype.getFieldType(i));
    }

    llvm_unreachable("Impossible state");
    return 0;
}

FieldPath::IndexingResult FieldPath::indexInto(P4HIR::IndexableTypeInterface itype,
                                               unsigned fieldID) {
    assert((fieldID > 0 && fieldID <= getMaxFieldID(itype)) && "Invalid field ID");
    unsigned fid = 0;
    unsigned i = 0;
    unsigned e = itype.getFieldCount();

    for (; i < e; i++) {
        // Move `fid` to the fieldID of the current field.
        fid += 1;

        // Compute the fieldID of the next field.
        unsigned nextFid = fid + getMaxFieldID(itype.getFieldType(i));

        // `fieldID` is in [fid, nextFid), we found what we're looking.
        if (fieldID < nextFid) return IndexingResult(i, fid, fieldID - fid);

        // Advance to next field.
        fid = nextFid;
    }

    llvm_unreachable("Impossible state");
    return IndexingResult(0, 0, 0);
}

FieldPath FieldPath::fromFieldID(mlir::Type rootType, unsigned fieldID) {
    assert((fieldID < getMaxFieldID(rootType)) && "Invalid fieldID");
    mlir::Type type = rootType;
    unsigned rhsFid = fieldID;

    while (rhsFid) {
        auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(type);
        auto indexingResult = indexInto(itype, rhsFid);
        type = itype.getFieldType(indexingResult.index);
        rhsFid = indexingResult.childFieldId;
    }

    return FieldPath(rootType, type, fieldID);
}

P4HIR::FieldPath FieldPath::concat(const FieldPath &lhs, const FieldPath &rhs) {
    assert(canConcat(lhs, rhs) && "Types cannot be concatenated");
    if (lhs.isEmpty()) return rhs;
    if (rhs.isEmpty()) return lhs;
    return FieldPath(lhs.getRootType(), rhs.getLeafType(), lhs.getFieldID() + rhs.getFieldID());
}

std::pair<FieldPath, FieldPath> FieldPath::split(llvm::function_ref<bool(FieldPath)> pred) const {
    FieldPath lhs;
    bool notFound = forEach([&](FieldPath prefixPath) {
        if (pred(prefixPath)) return false;

        lhs = prefixPath;
        return true;
    });

    if (notFound) return {*this, FieldPath()};
    if (lhs.isEmpty()) return {FieldPath(), *this};

    return {lhs, FieldPath(lhs.getLeafType(), getLeafType(), getFieldID() - lhs.getFieldID())};
}

FieldPath &FieldPath::append(unsigned index) {
    auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(leafType);
    fieldID += getFieldID(itype, index);
    leafType = itype.getFieldType(index);
    return *this;
}

bool FieldPath::forEach(llvm::function_ref<bool(FieldPath)> cb) const {
    // Iteration invariant: (lhsFid + rhsFid) == getFieldID()
    unsigned lhsFid = 0;
    unsigned rhsFid = getFieldID();
    mlir::Type type = getRootType();

    // Visit root.
    if (!cb(FieldPath(getRootType(), type, lhsFid))) return false;

    while (rhsFid) {
        auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(type);
        auto indexingResult = indexInto(itype, rhsFid);
        type = itype.getFieldType(indexingResult.index);

        if (!cb(FieldPath(getRootType(), type, lhsFid))) return false;

        lhsFid += indexingResult.parentFieldId;
        rhsFid = indexingResult.childFieldId;
    }

    return true;
}

bool FieldPath::forEachField(llvm::function_ref<bool(IndexedField)> cb) const {
    mlir::Type type = getRootType();
    unsigned rhsFid = getFieldID();

    while (rhsFid) {
        auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(type);
        auto indexingResult = indexInto(itype, rhsFid);

        if (!cb(IndexedField(itype, indexingResult.index))) return false;

        rhsFid = indexingResult.childFieldId;
    }

    return true;
}

std::string FieldPath::getIdentifier() const {
    llvm::SmallString<64> name;
    forEachField([&](auto field) {
        name += ".";
        name += field.getName();
        return true;
    });
    return (std::string)name;
}

void FieldPath::print(llvm::raw_ostream &os) const { os << getIdentifier(); }
