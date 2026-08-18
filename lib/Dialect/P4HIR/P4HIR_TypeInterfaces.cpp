// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;

mlir::Type IndexedField::getType() const {
    return mlir::cast<P4HIR::IndexableTypeInterface>(indexableType).getFieldType(index);
}

std::string IndexedField::getName() const {
    return mlir::cast<P4HIR::IndexableTypeInterface>(indexableType).getFieldName(index);
}

mlir::StringAttr IndexedField::getNameAttr() const {
    return mlir::cast<P4HIR::IndexableTypeInterface>(indexableType).getFieldNameAttr(index);
}

mlir::DictionaryAttr IndexedField::getAnnotations() const {
    return mlir::cast<P4HIR::IndexableTypeInterface>(indexableType).getFieldAnnotations(index);
}

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.cpp.inc"
