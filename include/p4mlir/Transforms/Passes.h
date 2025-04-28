#ifndef P4MLIR_TRANSFORMS_PASSES_H
#define P4MLIR_TRANSFORMS_PASSES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <memory>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

namespace P4::P4MLIR {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_SIMPLIFYPARSERS
#include "p4mlir/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createPrintParsersGraphPass();
std::unique_ptr<mlir::Pass> createSimplifyParsersPass();

#define GEN_PASS_REGISTRATION
#include "p4mlir/Transforms/Passes.h.inc"

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_PASSES_H
