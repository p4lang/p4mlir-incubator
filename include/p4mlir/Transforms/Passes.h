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
#define GEN_PASS_DECL_SERENUMELIMINATION
#define GEN_PASS_DECL_REMOVEALIASES
#define GEN_PASS_DECL_ENUMELIMINATION
#define GEN_PASS_DECL_REMOVESOFTCF
#define GEN_PASS_DECL_INLINEPARSERS
#define GEN_PASS_DECL_INLINECONTROLS
#include "p4mlir/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createPrintParsersGraphPass();
std::unique_ptr<mlir::Pass> createSimplifyParsersPass();
std::unique_ptr<mlir::Pass> createFlattenCFGPass();
std::unique_ptr<mlir::Pass> createSerEnumEliminationPass();
std::unique_ptr<mlir::Pass> createRemoveAliasesPass();
std::unique_ptr<mlir::Pass> createTupleToStructPass();
std::unique_ptr<mlir::Pass> createEnumEliminationPass();
std::unique_ptr<mlir::Pass> createRemoveSoftCFPass();
std::unique_ptr<mlir::Pass> createCopyInCopyOutEliminationPass();
std::unique_ptr<mlir::Pass> createInlineParsersPass();
std::unique_ptr<mlir::Pass> createInlineControlsPass();

#define GEN_PASS_REGISTRATION
#include "p4mlir/Transforms/Passes.h.inc"

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_PASSES_H
