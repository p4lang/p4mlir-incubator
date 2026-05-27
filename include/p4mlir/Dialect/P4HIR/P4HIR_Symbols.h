#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

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

/// A variant of 'lookupSymbol' that returns all of the symbols referenced by a
/// given SymbolRefAttr (including nested ones).  Returns failure if any of the
/// nested references could not be resolved.
mlir::LogicalResult lookupSymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *source,
                                 mlir::SymbolRefAttr symbol,
                                 llvm::SmallVectorImpl<mlir::Operation *> &symbols);

mlir::LogicalResult lookupSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol,
                                 llvm::SmallVectorImpl<mlir::Operation *> &symbols);

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_SYMBOLS_H
