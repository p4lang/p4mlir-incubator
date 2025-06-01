#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"

#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"

using namespace mlir;
using namespace P4::P4MLIR;

void P4CoreLib::P4CoreLibDialect::initialize() {
    registerTypes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.cpp.inc"  // NOLINT
        >();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.cpp.inc"  // NOLINT
