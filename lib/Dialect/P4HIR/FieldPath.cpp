// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Dialect/P4HIR/FieldPath.h"

#include "llvm/ADT/SmallString.h"

using namespace P4::P4MLIR;

using P4HIR::FieldIDs;
using P4HIR::FieldPath;

unsigned FieldIDs::getFieldID(P4HIR::IndexableTypeInterface itype, unsigned index) {
    assert((index < itype.getFieldCount()) && "Invalid field index");

    for (auto [field, fieldID] : getFieldsWithIDs(itype))
        if (field.getIndex() == index) return fieldID;

    llvm_unreachable("Impossible state");
    return 0;
}

FieldIDs::IndexingResult FieldIDs::indexInto(P4HIR::IndexableTypeInterface itype,
                                             unsigned fieldID) {
    assert(FieldIDs::isValidField(itype, fieldID) && "Invalid field ID");
    unsigned fid = 1;
    for (unsigned i = 0, e = itype.getFieldCount(); i < e; i++) {
        unsigned nextFid = FieldIDs::getNextFieldID(fid, itype, i);

        // `fieldID` is in [fid, nextFid), we found what we're looking.
        if (fieldID < nextFid) return IndexingResult(i, fid, fieldID - fid);

        // Advance to next field.
        fid = nextFid;
    }

    llvm_unreachable("Impossible state");
    return IndexingResult(0, 0, 0);
}

FieldPath FieldPath::fromFieldID(mlir::Type rootType, unsigned fieldID) {
    assert(FieldIDs::isValid(rootType, fieldID) && "Invalid fieldID");
    mlir::Type type = rootType;
    unsigned rhsFid = fieldID;

    while (rhsFid) {
        auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(type);
        auto indexingResult = FieldIDs::indexInto(itype, rhsFid);
        type = itype.getFieldType(indexingResult.index);
        rhsFid = indexingResult.childFieldId;
    }

    return FieldPath(rootType, type, fieldID);
}

static void forEachFieldPathImpl(FieldPath path, llvm::function_ref<void(FieldPath)> cb) {
    cb(path);

    auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(path.getType());
    if (!itype) return;

    for (auto field : itype.getFields())
        forEachFieldPathImpl(P4HIR::FieldPath::concat(path, field), cb);
}

void FieldPath::forEachFieldPath(mlir::Type type, llvm::function_ref<void(FieldPath)> cb) {
    forEachFieldPathImpl(FieldPath(type), cb);
}

P4HIR::FieldPath &FieldPath::concat(FieldPath rhs) {
    assert(canConcat(*this, rhs) && "Types cannot be concatenated");

    if (isEmpty()) {
        *this = rhs;
    } else if (!rhs.isEmpty()) {
        leafType = rhs.getType();
        fieldID += rhs.getFieldID();
    }

    return *this;
}

std::pair<FieldPath, FieldPath> FieldPath::split(llvm::function_ref<bool(FieldPath)> pred) const {
    FieldPath lhs;
    bool doSplit = false;
    for (auto prefixPath : iterPaths()) {
        lhs = prefixPath;
        if (pred(prefixPath)) {
            doSplit = true;
            break;
        }
    }

    if (!doSplit) return {*this, FieldPath()};
    if (lhs.isEmpty()) return {FieldPath(), *this};

    return {lhs, FieldPath(lhs.getType(), getType(), getFieldID() - lhs.getFieldID())};
}

bool FieldPath::try_append(unsigned index) {
    if (auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(leafType)) {
        assert((index < itype.getFieldCount()) && "Out of bounds index");
        fieldID += FieldIDs::getFieldID(itype, index);
        leafType = itype.getFieldType(index);
        return true;
    }

    return false;
}

bool FieldPath::try_append(llvm::StringRef name) {
    if (auto stype = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(leafType)) {
        if (auto field = stype.getFieldByName(name)) {
            fieldID += FieldIDs::getFieldID(stype, field->getIndex());
            leafType = field->getType();
            return true;
        }
    }

    return false;
}

FieldPath FieldPath::withRoot(mlir::Type newRootType) const {
    if (isEmpty()) return *this;

    FieldPath newPath(newRootType);
    for (auto field : iterFields()) newPath.append(field.getIndex());

    return newPath;
}

std::string FieldPath::getIdentifier(llvm::StringRef delimiter) const {
    llvm::SmallString<64> name;
    for (auto field : iterFields()) {
        if (!name.empty()) name += delimiter;
        name += field.getName();
    }
    return (std::string)name;
}

void FieldPath::print(llvm::raw_ostream &os) const { os << getIdentifier(); }
