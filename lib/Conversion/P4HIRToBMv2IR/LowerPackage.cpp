#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "p4hir-convert-to-bmv2"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERPACKAGE
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

struct LowerToV1SwitchPattern : public OpRewritePattern<P4HIR::InstantiateOp> {
    using OpRewritePattern<P4HIR::InstantiateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(P4HIR::InstantiateOp instOp,
                                  PatternRewriter &rewriter) const override {
        constexpr unsigned numV1SwitchArgs = 6;
        auto operands = instOp.getArgOperands();
        if (operands.size() != numV1SwitchArgs) return failure();
        auto loc = instOp.getLoc();
        SmallVector<Operation *>
            eraseList;  // For construct ops that should be unused after conversion
        auto parser = getSymRefAndRename<P4HIR::ParserType>(loc, operands[0], eraseList, "parser");
        if (failed(parser)) return failure();
        auto verifyChecksum = getSymRef<P4HIR::ControlType>(loc, operands[1], eraseList);
        if (failed(verifyChecksum)) return failure();
        auto ingress =
            getSymRefAndRename<P4HIR::ControlType>(loc, operands[2], eraseList, "ingress");
        if (failed(ingress)) return failure();
        auto egress = getSymRefAndRename<P4HIR::ControlType>(loc, operands[3], eraseList, "egress");
        if (failed(egress)) return failure();
        auto computeChecksum = getSymRef<P4HIR::ControlType>(loc, operands[4], eraseList);
        if (failed(computeChecksum)) return failure();
        auto deparser =
            getSymRefAndRename<P4HIR::ControlType>(loc, operands[5], eraseList, "deparser");
        if (failed(deparser)) return failure();

        rewriter.replaceOpWithNewOp<BMv2IR::V1SwitchOp>(
            instOp, instOp.getSymName(), parser.value(), verifyChecksum.value(), ingress.value(),
            egress.value(), computeChecksum.value(), deparser.value());
        for (auto op : eraseList) rewriter.eraseOp(op);
        return success();
    }

 private:
    template <typename Ty>
    static FailureOr<SymbolRefAttr> getSymRefAndRename(Location loc, Value val,
                                                       SmallVector<Operation *> &eraseList,
                                                       StringRef newName) {
        if (!isa<Ty>(val.getType())) return emitError(loc, "Unexpected type for argument");
        auto defOp = val.getDefiningOp<P4HIR::ConstructOp>();
        if (!defOp) return emitError(loc, "Expected defining operation to be a ConstructOp");
        eraseList.push_back(defOp);
        auto symRef = defOp.getCalleeAttr();
        auto moduleOp = defOp->getParentOfType<ModuleOp>();
        if (!moduleOp) return defOp->emitError("Expected module parent");
        auto oldNameAttr = symRef.getLeafReference();
        auto newNameAttr = StringAttr::get(defOp->getContext(), newName);
        auto symDefOp =
            dyn_cast_or_null<SymbolOpInterface>(SymbolTable::lookupSymbolIn(moduleOp, oldNameAttr));
        if (!symDefOp) return defOp.emitError("Expected defining symbol");
        auto checkNewNameOp = SymbolTable::lookupSymbolIn(moduleOp, newNameAttr);
        if (checkNewNameOp && checkNewNameOp != symDefOp)
            return checkNewNameOp->emitError("Symbol already defined");
        symDefOp.setName(newName);
        // Replace all symbol references in the nested region
        if (failed(SymbolTable::replaceAllSymbolUses(oldNameAttr, newNameAttr, symDefOp)))
            return failure();
        // Replace all references in the module
        if (failed(SymbolTable::replaceAllSymbolUses(oldNameAttr, newNameAttr, moduleOp)))
            return failure();
        return defOp.getCalleeAttr();
    }

    template <typename Ty>
    static FailureOr<SymbolRefAttr> getSymRef(Location loc, Value val,
                                              SmallVector<Operation *> &eraseList) {
        if (!isa<Ty>(val.getType())) return emitError(loc, "Unexpected type for argument");
        auto defOp = val.getDefiningOp<P4HIR::ConstructOp>();
        if (!defOp) return emitError(loc, "Expected defining operation to be a ConstructOp");
        eraseList.push_back(defOp);
        return defOp.getCalleeAttr();
    }
};

struct LowerPackagePass : public P4::P4MLIR::impl::LowerPackageBase<LowerPackagePass> {
    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp module = getOperation();
        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        patterns.add<LowerToV1SwitchPattern>(&context);

        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<P4HIR::InstantiateOp>();
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
}  // anonymous namespace
