#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "p4mlir/Dialect/P4HIR/P4HIROps.h"
#include "p4mlir/Dialect/BMv2/BMv2Ops.h"

using namespace mlir;

namespace {
struct LowerV1SwitchPattern
    : public OpRewritePattern<p4hir::InstantiateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      p4hir::InstantiateOp op,
      PatternRewriter &rewriter) const override {

    if (op.getTargetName() != "V1Switch")
      return failure();

    auto operands = op.getOperands();

    rewriter.replaceOpWithNewOp<p4mlir::bmv2::V1SwitchOp>(
        op,
        operands[0], // parser
        operands[1], // verify
        operands[2], // ingress
        operands[3], // egress
        operands[4], // update
        operands[5]  // deparser
    );

    return success();
  }
};
}

struct LowerP4HIRToBMv2Pass
    : public PassWrapper<LowerP4HIRToBMv2Pass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerV1SwitchPattern>(&getContext());
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createLowerP4HIRToBMv2Pass() {
  return std::make_unique<LowerP4HIRToBMv2Pass>();
}
