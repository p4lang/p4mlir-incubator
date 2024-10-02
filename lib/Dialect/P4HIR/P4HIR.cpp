#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"

namespace P4::P4MLIR::P4HIR {

void P4HIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"
      >();
}

} // namespace P4::P4MLIR::P4HIR
