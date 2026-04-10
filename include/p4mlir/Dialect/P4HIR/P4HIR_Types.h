#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

namespace P4::P4MLIR::P4HIR {
mlir::TypedAttr getStructLikeDefaultValue(StructLikeTypeInterface type);

void printFieldInfo(mlir::AsmPrinter &p, const FieldInfo &fi);
mlir::ParseResult parseFieldInfo(mlir::AsmParser &p, mlir::FailureOr<FieldInfo> &fi);

/// Parse a list of unique field names and types within <> plus name. E.g.:
/// <name, foo: i7, bar: i8>
mlir::ParseResult parseFields(mlir::AsmParser &p, std::string &name,
                              llvm::SmallVectorImpl<FieldInfo> &parameters,
                              mlir::DictionaryAttr &annotations);

/// Print out a list of named fields surrounded by <>.
void printFields(mlir::AsmPrinter &p, llvm::StringRef name, llvm::ArrayRef<FieldInfo> fields,
                 mlir::DictionaryAttr annotations);
}  // namespace P4::P4MLIR::P4HIR

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_TYPES_H
