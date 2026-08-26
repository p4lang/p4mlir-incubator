// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

namespace P4::P4MLIR::P4HIR {

using FieldWithID = std::pair<IndexedField, unsigned>;

/// This struct provides various utilities for performing operations related to FieldIDs.
/// A FieldID is a unique DFS-based ID of a field nested somewhere within an indexable-type.
/// For example:
/// ```
/// struct a  /* 0 */ {
///   int b; /* 1 */
///   struct c /* 2 */ {
///     int d; /* 3 */
///   }
/// }
/// ```
struct FieldIDs {
 private:
    static unsigned getNextFieldID(unsigned currentFieldID, P4HIR::IndexableTypeInterface itype,
                                   unsigned index) {
        return currentFieldID + 1 + getMaxFieldID(itype.getFieldType(index));
    }

    /// An iterator type to iterate fields together with their fieldIDs.
    struct FieldWithIDIterator
        : public llvm::iterator_facade_base<FieldWithIDIterator, std::forward_iterator_tag,
                                            FieldWithID> {
        FieldWithIDIterator(P4HIR::IndexableTypeInterface type, unsigned index, unsigned fieldID)
            : type(type), index(index), fieldID(fieldID) {}

        FieldWithID operator*() const { return {type.getField(index), fieldID}; }

        bool operator==(const FieldWithIDIterator &other) const {
            assert((type == other.type) && "Invariant");
            return index == other.index;
        }

        using llvm::iterator_facade_base<FieldWithIDIterator, std::forward_iterator_tag,
                                         FieldWithID>::operator++;

        void operator++() {
            fieldID = getNextFieldID(fieldID, type, index);
            index++;
        }

     private:
        P4HIR::IndexableTypeInterface type;
        unsigned index;
        unsigned fieldID;
    };

 public:
    /// Return a range to iterate the fields in `itype` paired with their fieldIDs.
    static mlir::iterator_range<FieldWithIDIterator> getFieldsWithIDs(
        P4HIR::IndexableTypeInterface itype) {
        return {FieldWithIDIterator(itype, 0, 1),
                FieldWithIDIterator(itype, itype.getFieldCount(), 0)};
    }

    /// Get the max fieldID that can reference a field of `type`.
    static unsigned getMaxFieldID(mlir::Type type) {
        if (auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(type))
            return itype.getNestedFieldCount();

        return 0;
    }

    /// Return true if `fieldID` is within the valid range for `type`.
    static bool isValid(mlir::Type type, unsigned fieldID) {
        return fieldID <= getMaxFieldID(type);
    }

    /// Return true if `fieldID` is within the valid range for to represent a field of `type`.
    static bool isValidField(mlir::Type type, unsigned fieldID) {
        return fieldID > 0 && fieldID <= getMaxFieldID(type);
    }

    /// Get the fieldID for the field at `index` in `itype`.
    static unsigned getFieldID(P4HIR::IndexableTypeInterface itype, unsigned index);

    /// The results of indexing in an IndexableType using a fieldID.
    struct IndexingResult {
        IndexingResult(unsigned index, unsigned parentFieldId, unsigned childFieldId)
            : index(index), parentFieldId(parentFieldId), childFieldId(childFieldId) {}

        /// The index of the field in the IndexableType type that corresponds to the given fieldID.
        unsigned index;
        /// The fieldID for the field at `index`.
        unsigned parentFieldId;
        /// The sub-fieldID to index within the resulting field.
        unsigned childFieldId;
    };

    /// Find the immediate field of `itype` that contains the field referenced by `fieldID` and
    /// return indexing information.
    static IndexingResult indexInto(P4HIR::IndexableTypeInterface itype, unsigned fieldID);
};

/// Represents a field path starting from a root type.
struct FieldPath {
 private:
    explicit FieldPath(mlir::Type rootType, mlir::Type leafType, unsigned fieldID)
        : rootType(rootType), leafType(leafType), fieldID(fieldID) {}

 public:
    /// Construct a special empty FieldPath.
    explicit FieldPath() : FieldPath(mlir::Type(), mlir::Type(), 0) {}

    /// Construct a field path representing `rootType`.
    explicit FieldPath(mlir::Type rootType) : FieldPath(rootType, rootType, 0) {}

    /// Construct a field path from a IndexedField. Allow implicit construction.
    FieldPath(P4HIR::IndexedField field) : FieldPath(field.getParentType()) {
        append(field.getIndex());
    }

    /// Create a FieldPath by fully indexing into `rootType` using `fieldID`.
    static FieldPath fromFieldID(mlir::Type rootType, unsigned fieldID);

    /// Perform a preorder DFS traversal on all fields nested in `type` and call `cb`.
    static void forEachFieldPath(mlir::Type type, llvm::function_ref<void(FieldPath)> cb);

    /// Returns true if the given fields paths can be concatenated into a single path with `concat`.
    static bool canConcat(FieldPath lhs, FieldPath rhs) {
        return lhs.isEmpty() || rhs.isEmpty() || lhs.getType() == rhs.getRootType();
    }

    /// Given two fields paths X->...->Y and Y->...->Z returns X->...->Z.
    /// Returns FieldPath() if either pah is the special empty path.
    /// Assumes that `canConcat(lhs, rhs) == true`.
    static FieldPath concat(FieldPath lhs, FieldPath rhs) { return lhs.concat(rhs); }

    /// If this is a path X->...->Y and `rhs` is another path Y->...->Z returns X->...->Z.
    /// Returns FieldPath() if either pah is the special empty path.
    /// Assumes that `canConcat(lhs, rhs) == true`.
    FieldPath &concat(FieldPath rhs);

    /// If this path is of the form X->...->Y->...->Z where Y is the first prefix path for
    /// which `pred(Y)` is true then return two new field paths X->...->Y and Y->...->Z.
    /// If pred(X) is true, returns {FieldPath(), *this}
    /// If there exists no such Y, return {*this, FieldPath()}
    std::pair<FieldPath, FieldPath> split(llvm::function_ref<bool(FieldPath)> pred) const;

    /// If this is a path X->...->Y and Y is an indexable type with a field Z at `index` make this
    /// path represent X->...->Y->Z and return true. Otherwise return false;
    bool try_append(unsigned index);

    bool try_append(P4HIR::IndexedField field) {
        assert((getType() == field.getParentType()) && "Incorrect indexed field");
        return try_append(field.getIndex());
    }

    /// If this is a path X->...->Y and Y is a struct-like type with a field Z named `name` make
    /// this path represent X->...->Y->Z and return true. Otherwise return false;
    bool try_append(llvm::StringRef name);

    FieldPath &append(unsigned index) {
        [[maybe_unused]] bool success = try_append(index);
        assert(success && "Missing field");
        return *this;
    }

    FieldPath &append(P4HIR::IndexedField field) {
        assert((getType() == field.getParentType()) && "Incorrect indexed field");
        return append(field.getIndex());
    }

    FieldPath &append(llvm::StringRef name) {
        [[maybe_unused]] bool success = try_append(name);
        assert(success && "Missing field");
        return *this;
    }

    template <typename T>
    FieldPath operator[](T &&arg) const {
        FieldPath res = *this;
        res.append(std::forward<T>(arg));
        return res;
    }

    /// Return true is this is the special empty path.
    bool isEmpty() const { return *this == FieldPath(); }
    explicit operator bool() const { return !isEmpty(); }

    /// Return the root type of this path.
    mlir::Type getRootType() const { return rootType; }

    /// Return the leaf type of this path.
    mlir::Type getType() const { return leafType; }

    /// Return a unique ID for the referenced field within the root type.
    unsigned getFieldID() const { return fieldID; }

    // Get an identified for the the full path.
    std::string getIdentifier(llvm::StringRef delimiter = ".") const;

    bool operator==(const FieldPath &rhs) const {
        return getRootType() == rhs.getRootType() && getType() == rhs.getType() &&
               getFieldID() == rhs.getFieldID();
    }

    bool operator!=(const FieldPath &rhs) const { return !operator==(rhs); }

 private:
    // A flexible iterator type to iterate parts of a fieldID access in a type.
    using PathIteratorData = std::tuple<mlir::Type, mlir::Type, unsigned, unsigned, unsigned>;
    struct PathIterator : public llvm::iterator_facade_base<PathIterator, std::forward_iterator_tag,
                                                            PathIteratorData> {
        PathIterator(mlir::Type type, unsigned fieldID)
            : parentType(type), type(type), index(0), lhsFid(0), rhsFid(fieldID) {}

        static PathIterator end() { return PathIterator(mlir::Type(), 0); }

        PathIteratorData operator*() const { return {parentType, type, index, lhsFid, rhsFid}; }

        bool operator==(const PathIterator &other) const {
            return type == other.type && lhsFid == other.lhsFid && rhsFid == other.rhsFid;
        }

        using llvm::iterator_facade_base<PathIterator, std::forward_iterator_tag,
                                         PathIteratorData>::operator++;

        void operator++() {
            if (rhsFid == 0) {
                // Advance to special end iterator state.
                *this = end();
                return;
            }

            auto itype = mlir::cast<P4HIR::IndexableTypeInterface>(type);
            auto indexingResult = FieldIDs::indexInto(itype, rhsFid);
            parentType = type;
            type = itype.getFieldType(indexingResult.index);
            index = indexingResult.index;
            lhsFid += indexingResult.parentFieldId;
            rhsFid = indexingResult.childFieldId;
        }

     private:
        mlir::Type parentType;
        mlir::Type type;
        unsigned index;
        unsigned lhsFid;
        unsigned rhsFid;
    };

    PathIterator begin() const { return PathIterator(getRootType(), getFieldID()); }
    PathIterator end() const { return PathIterator::end(); }

 public:
    /// Iterate all prefix paths that make up this path.
    /// For a path T.x.y.z this will yield T, T.x, T.x.y and T.x.y.z.
    auto iterPaths() const {
        mlir::iterator_range<PathIterator> range = {begin(), end()};
        return llvm::map_range(range, [&](auto info) {
            auto [parentType, type, index, lhsFid, rhsFid] = info;
            return FieldPath(getRootType(), type, lhsFid);
        });
    }

    /// Iterate all fields that make up this path.
    /// For a path T.x.y.z this will yield .x, .y and .z.
    auto iterFields() const {
        mlir::iterator_range<PathIterator> range = {std::next(begin()), end()};
        return llvm::map_range(range, [&](auto info) {
            auto [parentType, type, index, lhsFid, rhsFid] = info;
            return IndexedField(parentType, index);
        });
    }

    /// Create a new path by appending the field indices of this path on top of `newRootType`.
    FieldPath withRoot(mlir::Type newRootType) const;

    void print(llvm::raw_ostream &os) const;

    std::string str() const {
        std::string buffer;
        llvm::raw_string_ostream ss(buffer);
        print(ss);
        return buffer;
    }

    struct DenseMapInfo {
        static inline FieldPath getEmptyKey() {
            FieldPath specialEmpty(mlir::Type(), mlir::Type(),
                                   std::numeric_limits<unsigned>::max());
            return specialEmpty;
        }

        static inline FieldPath getTombstoneKey() {
            FieldPath specialTombstone(mlir::Type(), mlir::Type(),
                                       std::numeric_limits<unsigned>::max() - 1);
            return specialTombstone;
        }

        static unsigned getHashValue(const FieldPath &path) {
            return llvm::hash_combine(path.getRootType(), path.getType(), path.getFieldID());
        }

        static bool isEqual(const FieldPath &lhs, const FieldPath &rhs) { return lhs == rhs; }
    };

 private:
    /// The root type that we're indexing into.
    mlir::Type rootType;
    /// Cached type of the referenced field.
    mlir::Type leafType;
    /// The fieldID that indexes into `rootType`.
    unsigned fieldID;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FieldPath &path) {
    path.print(os);
    return os;
}

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H
