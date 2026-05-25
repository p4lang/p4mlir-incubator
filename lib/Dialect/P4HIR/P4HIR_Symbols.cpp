#include "p4mlir/Dialect/P4HIR/P4HIR_Symbols.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace P4::P4MLIR;

static mlir::ModuleOp getParentModule(Operation *from) {
    if (auto moduleOp = dyn_cast<mlir::ModuleOp>(from)) return moduleOp;

    auto moduleOp = from->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) llvm_unreachable("could not find parent module op");

    return moduleOp;
}

static mlir::Operation *lookupGlobalSymbolImpl(
    mlir::Operation *source, mlir::SymbolRefAttr symbol,
    llvm::function_ref<mlir::Operation *(mlir::Operation *, mlir::StringAttr)> lookupSymbolFn) {
    // Global symbols must be fully qualified
    if (isa<mlir::FlatSymbolRefAttr>(symbol)) return nullptr;

    // Root reference should be module
    auto moduleOp = getParentModule(source);
    if (symbol.getRootReference() != moduleOp.getSymNameAttr()) return nullptr;

    // Lookup each of the nested references.
    mlir::Operation *symbolOp = moduleOp;
    for (mlir::FlatSymbolRefAttr ref : symbol.getNestedReferences()) {
        // Check that we have a valid symbol table to lookup ref.
        if (!symbolOp->hasTrait<OpTrait::SymbolTable>()) return nullptr;
        symbolOp = lookupSymbolFn(symbolOp, ref.getAttr());
        // If the nested symbol is private, lookup failed.
        if (!symbolOp
            // || SymbolTable::getSymbolVisibility(symbolOp) == SymbolTable::Visibility::Private
        )
            return nullptr;
    }

    return symbolOp;
}

mlir::Operation *P4HIR::lookupGlobalSymbol(mlir::SymbolTableCollection &symbolTable,
                                           mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    auto lookupFn = [&](Operation *symbolTableOp, StringAttr symbol) {
        return symbolTable.lookupSymbolIn(symbolTableOp, symbol);
    };

    return lookupGlobalSymbolImpl(source, symbol, lookupFn);
}

mlir::Operation *P4HIR::lookupGlobalSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    auto lookupFn = [&](Operation *symbolTableOp, StringAttr symbol) {
        return SymbolTable::lookupSymbolIn(symbolTableOp, symbol);
    };

    return lookupGlobalSymbolImpl(source, symbol, lookupFn);
}

mlir::Operation *P4HIR::lookupSymbol(mlir::SymbolTableCollection &symbolTable,
                                     mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    // Local symbols, resolved against local symbol table
    if (isa<mlir::FlatSymbolRefAttr>(symbol))
        return symbolTable.lookupNearestSymbolFrom(source, symbol);

    return lookupGlobalSymbol(symbolTable, source, symbol);
}

mlir::Operation *P4HIR::lookupSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    // Local symbols, resolved against local symbol table
    if (isa<mlir::FlatSymbolRefAttr>(symbol))
        return SymbolTable::lookupNearestSymbolFrom(source, symbol);

    return lookupGlobalSymbol(source, symbol);
}
