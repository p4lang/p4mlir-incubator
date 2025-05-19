//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_PASSES_H
#define P4MLIR_CONVERSION_PASSES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "p4mlir/Conversion/P4HIRToCoreLib.h"

namespace P4::P4MLIR {

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "p4mlir/Conversion/Passes.h.inc"

}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_PASSES_H
