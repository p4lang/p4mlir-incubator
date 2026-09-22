#include "p4mlir/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

using namespace mlir;

namespace P4::P4MLIR {

#define GEN_PASS_DEF_EVALUATESTATICASSERT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct EvaluateStaticAssertPass
    : public impl::EvaluateStaticAssertBase<EvaluateStaticAssertPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool hasError = false;

    SmallVector<P4CoreLib::StaticAssertOp> assertOps;
    moduleOp->walk([&](P4CoreLib::StaticAssertOp op) {
      assertOps.push_back(op);
    });

    for (auto op : assertOps) {
      Value cond = op.getCond();

      auto constOp = cond.getDefiningOp<P4HIR::ConstOp>();
      if (!constOp)
        continue;

      auto boolAttr = llvm::dyn_cast<P4HIR::BoolAttr>(constOp.getValue());
      if (!boolAttr)
        continue;

      if (boolAttr.getValue()) {
        OpBuilder builder(op);
        auto trueAttr = P4HIR::BoolAttr::get(op.getContext(), true);
        auto replacement = builder.create<P4HIR::ConstOp>(op.getLoc(), trueAttr);
        op.replaceAllUsesWith(replacement.getResult());
        op.erase();
      } else {
        if (auto msg = op.getMessage())
          op.emitError() << "static assertion failed: " << msg.value();
        else
          op.emitError("static assertion failed");
        hasError = true;
      }
    }

    if (hasError)
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createEvaluateStaticAssertPass() {
  return std::make_unique<EvaluateStaticAssertPass>();
}

}  // namespace P4::P4MLIR
