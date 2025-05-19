#ifndef P4MLIR_CONVERSION_P4HIRTOCORELIB_H
#define P4MLIR_CONVERSION_P4HIRTOCORELIB_H

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <memory>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace P4::P4MLIR {

#define GEN_PASS_DECL_LOWERTOP4CORELIB
#include "p4mlir/Conversion/Passes.h.inc"
}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_P4HIRTOCORELIB_H
