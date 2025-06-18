
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace P4::P4MLIR::P4CoreLib;

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.cpp.inc"

void P4CoreLibDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.cpp.inc"  // NOLINT
        >();
}
