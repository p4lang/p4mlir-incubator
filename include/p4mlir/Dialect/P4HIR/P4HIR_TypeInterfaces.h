#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace P4::P4MLIR::P4HIR {

/// Struct defining a field. Used in structs and header
struct FieldInfo {
    mlir::StringAttr name;
    mlir::Type type;
    mlir::DictionaryAttr annotations;

  FieldInfo(mlir::StringAttr name,
            mlir::Type type,
            mlir::DictionaryAttr annotations = {})
    : name(name), type(type),
      annotations(annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr())
  { }
};

namespace FieldIdImpl {
unsigned getMaxFieldID(::mlir::Type);

std::pair<::mlir::Type, unsigned> getSubTypeByFieldID(::mlir::Type, unsigned fieldID);

::mlir::Type getFinalTypeByFieldID(::mlir::Type type, unsigned fieldID);

std::pair<unsigned, bool> projectToChildFieldID(::mlir::Type, unsigned fieldID, unsigned index);

std::pair<unsigned, unsigned> getIndexAndSubfieldID(::mlir::Type type, unsigned fieldID);

unsigned getFieldID(::mlir::Type type, unsigned index);

unsigned getIndexForFieldID(::mlir::Type type, unsigned fieldID);

}  // namespace FieldIdImpl
}  // namespace P4::P4MLIR::P4HIR

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_H
