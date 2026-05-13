//===- P4HIRSymbols.h - P4HIR-related symbol logic --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages lookup logic for P4HIR absolute symbols.
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

namespace P4::P4MLIR::P4HIR {

/// Looks up a symbol from the global module symbol table.
/// This exploits SymbolTableCollection for better symbol table lookup.
mlir::Operation *lookupGlobalSymbol(mlir::SymbolTableCollection &symbolTable,
                                    mlir::Operation *source, mlir::SymbolRefAttr symbol);
template <class T>
T lookupGlobalSymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *source,
                     mlir::SymbolRefAttr symbol) {
    return mlir::dyn_cast_or_null<T>(lookupGlobalSymbol(symbolTable, source, symbol));
}

/// Looks up a symbol from the global module symbol table.
mlir::Operation *lookupGlobalSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol);
template <class T>
T lookupGlobalSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    return mlir::dyn_cast_or_null<T>(lookupGlobalSymbol(source, symbol));
}

/// Same as above, but performs local or global lookup depending on the symbol kind
mlir::Operation *lookupSymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *source,
                              mlir::SymbolRefAttr symbol);
template <class T>
T lookupSymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *source,
               mlir::SymbolRefAttr symbol) {
    return mlir::dyn_cast_or_null<T>(lookupSymbol(symbolTable, source, symbol));
}

mlir::Operation *lookupSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol);
template <class T>
T lookupSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    return mlir::dyn_cast_or_null<T>(lookupSymbol(source, symbol));
}

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H
