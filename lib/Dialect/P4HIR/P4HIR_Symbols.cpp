#include "p4mlir/Dialect/P4HIR/P4HIR_Symbols.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace P4::P4MLIR;

static mlir::ModuleOp getParentModule(Operation *from) {
    auto moduleOp = from->getParentOfType<mlir::ModuleOp>();

    if (!moduleOp) llvm_unreachable("could not find parent module op");

    return moduleOp;
}

mlir::Operation *P4HIR::lookupGlobalSymbol(mlir::SymbolTableCollection &symbolTable,
                                           mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    // Global symbols must be fully qualified
    if (isa<mlir::FlatSymbolRefAttr>(symbol) || symbol.getNestedReferences().size() != 1)
        return nullptr;

    auto moduleOp = getParentModule(source);
    if (symbol.getRootReference() != moduleOp.getSymNameAttr()) return nullptr;

    return symbolTable.lookupSymbolIn(moduleOp, symbol.getLeafReference());
}

mlir::Operation *P4HIR::lookupGlobalSymbol(mlir::Operation *source, mlir::SymbolRefAttr symbol) {
    // Global symbols must be fully qualified
    if (isa<mlir::FlatSymbolRefAttr>(symbol) || symbol.getNestedReferences().size() != 1)
        return nullptr;

    auto moduleOp = getParentModule(source);
    if (symbol.getRootReference() != moduleOp.getSymNameAttr()) return nullptr;

    return SymbolTable::lookupSymbolIn(moduleOp, symbol.getLeafReference());
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
