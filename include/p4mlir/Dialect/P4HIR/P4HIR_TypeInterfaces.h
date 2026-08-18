// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace P4::P4MLIR::P4HIR {

/// Struct that represents a particular field of an IndexableType.
struct IndexedField {
    IndexedField(mlir::Type indexableType, unsigned index)
        : indexableType(indexableType), index(index) {}
    unsigned getIndex() const { return index; }
    mlir::Type getParentType() const { return indexableType; }
    mlir::Type getType() const;
    std::string getName() const;
    mlir::StringAttr getNameAttr() const;
    mlir::DictionaryAttr getAnnotations() const;

    bool operator==(const IndexedField &o) const {
        return indexableType == o.indexableType && index == o.index;
    }
    bool operator!=(const IndexedField &o) const { return !operator==(o); }

 private:
    mlir::Type indexableType;
    unsigned index;
};

/// Struct field definition. Used to define structs and headers
struct FieldInfo {
    FieldInfo(mlir::StringAttr name, mlir::Type type, mlir::DictionaryAttr annotations = {})
        : name(name),
          type(type),
          annotations(annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr()) {}

    mlir::Type getType() const { return type; }
    std::string getName() const { return name.getValue().str(); }
    mlir::StringAttr getNameAttr() const { return name; }
    mlir::DictionaryAttr getAnnotations() const { return annotations; }

    mlir::StringAttr name;
    mlir::Type type;
    mlir::DictionaryAttr annotations;
};

}  // namespace P4::P4MLIR::P4HIR

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H
