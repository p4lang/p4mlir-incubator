// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-use-controlplane-names"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_USECONTROLPLANENAMES
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct TableOpRenamePattern : public OpRewritePattern<P4HIR::TableOp> {
    using OpRewritePattern<P4HIR::TableOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::TableOp tableOp,
                                  PatternRewriter &rewriter) const override {
        auto controlPlaneName = llvm::dyn_cast_or_null<StringAttr>(tableOp.getAnnotation("name"));
        if (!controlPlaneName) return tableOp.emitError("No control plane name");
        auto control = tableOp->getParentOfType<P4HIR::ControlOp>();
        if (!control) return tableOp.emitError("Expected control parent");
        auto moduleOp = tableOp->getParentOfType<ModuleOp>();
        auto newRef =
            SymbolRefAttr::get(control.getSymNameAttr(), {SymbolRefAttr::get(controlPlaneName)});
        control.walk([&](P4HIR::TableApplyOp applyOp) {
            auto tableRef = applyOp.getTable();
            auto calledTable = SymbolTable::lookupSymbolIn(moduleOp, tableRef);
            if (calledTable != tableOp.getOperation()) return WalkResult::skip();
            applyOp.setTableAttr(newRef);
            return WalkResult::advance();
        });

        tableOp.setSymNameAttr(controlPlaneName);
        return success();
    }
};

struct UseControlPlaneNamesPass
    : public P4::P4MLIR::impl::UseControlPlaneNamesBase<UseControlPlaneNamesPass> {
    UseControlPlaneNamesPass() = default;
    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();
        MLIRContext &context = getContext();

        RewritePatternSet patterns(&context);
        ConversionTarget target(context);

        target.addDynamicallyLegalOp<P4HIR::TableOp>([](P4HIR::TableOp tableOp) {
            auto nameAttr = tableOp.getAnnotation("name");
            if (!nameAttr) return true;
            auto controlPlaneName = dyn_cast_or_null<StringAttr>(tableOp.getAnnotation("name"));
            if (!controlPlaneName) return true;
            return controlPlaneName.getValue() == tableOp.getSymName();
        });
        patterns.add<TableOpRenamePattern>(&context);
        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
            signalPassFailure();
    }
};

}  // end namespace

std::unique_ptr<Pass> createUseControlPlaneNamesPass() {
    return std::make_unique<UseControlPlaneNamesPass>();
}

}  // namespace P4::P4MLIR
