// Strength reduction pass using PDLL patterns

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include "StrengthReductionPatterns.h.inc"

namespace P4 {
namespace P4MLIR {

namespace {
struct StrengthReductionPass
    : public mlir::PassWrapper<StrengthReductionPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrengthReductionPass)

  llvm::StringRef getArgument() const final { return "p4hir-strength-reduction"; }
  llvm::StringRef getDescription() const final {
    return "Apply strength reduction patterns to P4HIR operations";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::pdl::PDLDialect, mlir::pdl_interp::PDLInterpDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert(MulToShl(&getContext()));
    patterns.insert(DivToShr(&getContext()));
    patterns.insert(ModToAnd(&getContext()));

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createP4HIRStrengthReductionPass() {
  return std::make_unique<StrengthReductionPass>();
}

void registerP4HIRStrengthReductionPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createP4HIRStrengthReductionPass();
  });
}

} // namespace P4MLIR
} // namespace P4
