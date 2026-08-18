// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

namespace P4::P4MLIR::P4HIR {

/// Represents a field path starting from a root type.
struct FieldPath {
 private:
    explicit FieldPath(mlir::Type rootType, mlir::Type leafType, unsigned fieldID)
        : rootType(rootType), leafType(leafType), fieldID(fieldID) {}

 public:
    /// Get the max fieldID that reference a field of `type`.
    static unsigned getMaxFieldID(mlir::Type type);

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

    /// Find the immidiate field of `itype` that contains the field referenced by `fieldID` and
    /// return indexing information.
    static IndexingResult indexInto(P4HIR::IndexableTypeInterface itype, unsigned fieldID);

    /// Create a FieldPath by fully indexing into `rootType` using `fieldID`.
    static FieldPath fromFieldID(mlir::Type rootType, unsigned fieldID);

    /// Construct a special empty FieldPath.
    explicit FieldPath() : FieldPath(mlir::Type(), mlir::Type(), 0) {}

    /// Construct a field path representing `rootType`.
    explicit FieldPath(mlir::Type rootType) : FieldPath(rootType, rootType, 0) {}

    /// Returns true if the given fields paths can be concatenated into a single path with `concat`.
    static bool canConcat(const FieldPath &lhs, const FieldPath &rhs) {
        return lhs.isEmpty() || rhs.isEmpty() || lhs.getLeafType() == rhs.getRootType();
    }

    /// Given two fields paths X->...->Y and Y->...->Z returns X->...->Z.
    /// Returns FieldPath() if either pah is the special empty path.
    /// Assumes that `canConcat(lhs, rhs) == true`.
    static FieldPath concat(const FieldPath &lhs, const FieldPath &rhs);

    /// If this path is of the form X->...->Y->...->Z where Y is the first field for
    /// which `pred(Y)` is true then return two new field paths X->...->Y and Y->...->Z.
    /// If pred(X) is true, returns {FieldPath(), *this}
    /// If there exists no such Y, return {*this, FieldPath()}
    std::pair<FieldPath, FieldPath> split(llvm::function_ref<bool(FieldPath)> pred) const;

    /// If this is a path X->...->Y and Y is an indexable type with a field Z at `index` make this
    /// path represent X->...->Y->Z.
    FieldPath &append(unsigned index);

    FieldPath operator[](unsigned index) {
        FieldPath res = *this;
        res.append(index);
        return res;
    }

    /// Return true is this is the special empty path.
    bool isEmpty() const { return *this == FieldPath(); }

    mlir::Type getRootType() const { return rootType; }
    mlir::Type getLeafType() const { return leafType; }

    /// Return a unique ID for the referenced field within the root type.
    unsigned getFieldID() const { return fieldID; }

    // Get an identified for the the full path.
    std::string getIdentifier() const;

    bool operator==(const FieldPath &rhs) const {
        return getRootType() == rhs.getRootType() && getLeafType() == rhs.getLeafType() &&
               getFieldID() == rhs.getFieldID();
    }

    bool operator!=(const FieldPath &rhs) const { return !operator==(rhs); }

    /// Helper function to iterate all prefix FieldPaths from root down to leaf.
    /// Returning false from `cb` will stop the iteration.
    bool forEach(llvm::function_ref<bool(FieldPath)> cb) const;

    /// Helper function to iterate all IndexedFields from root down to leaf.
    /// Returning false from `cb` will stop the iteration.
    bool forEachField(llvm::function_ref<bool(IndexedField)> cb) const;

    void print(llvm::raw_ostream &os) const;

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
            return llvm::hash_combine(path.getRootType(), path.getLeafType(), path.getFieldID());
        }

        static bool isEqual(const FieldPath &lhs, const FieldPath &rhs) { return lhs == rhs; }
    };

 private:
    /// The root type that we're indexing into.
    mlir::Type rootType;
    /// Cached type of the referenced field.
    mlir::Type leafType;
    /// FieldID is a unique DFS-based ID of the referenced field within `rootType`. For example:
    /// ```
    /// struct a  /* 0 */ {
    ///   int b; /* 1 */
    ///   struct c /* 2 */ {
    ///     int d; /* 3 */
    ///   }
    /// }
    /// ```
    unsigned fieldID;
};

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_FIELDPATH_H
