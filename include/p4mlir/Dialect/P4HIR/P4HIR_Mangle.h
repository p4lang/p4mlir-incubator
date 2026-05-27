// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

namespace P4::P4MLIR::P4HIR {

class Mangler {
 public:
    void getName(llvm::raw_ostream &os, mlir::Type type) const;
    void getName(llvm::SmallVectorImpl<char> &buf, mlir::Type type) const;
    [[nodiscard]] mlir::StringAttr getName(mlir::Type type) const;

    // Same as above, but for functions / methods - besides type also encodes parameter names (!)
    [[nodiscard]] mlir::StringAttr getFunctionName(
        P4HIR::FuncType type, llvm::ArrayRef<mlir::DictionaryAttr> paramAttrs) const;
    [[nodiscard]] mlir::StringAttr getFunctionName(P4HIR::FuncOp op) const;
    // Provide mangled name for extern instantiation
    [[nodiscard]] mlir::StringAttr getExternName(P4HIR::ExternType type,
                                                 llvm::ArrayRef<mlir::Type> typeArguments) const;
};

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_MANGLE_H
