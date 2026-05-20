#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace P4::P4MLIR::P4HIR {

class Mangler {
 public:
    void getName(llvm::raw_ostream &os, mlir::Type type) const;
    void getName(llvm::SmallVectorImpl<char> &buf, mlir::Type type) const;
    [[nodiscard]] mlir::StringAttr getName(mlir::Type type) const;
};

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H
