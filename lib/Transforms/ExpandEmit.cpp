#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"
#define DEBUG_TYPE "p4hir-expand-emit"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_EXPANDEMIT
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct ExpandEmitPass : public P4::P4MLIR::impl::ExpandEmitBase<ExpandEmitPass> {
    ExpandEmitPass() = default;
    void runOnOperation() override;
};

struct ExpandEmitPattern : public mlir::OpRewritePattern<P4CoreLib::PacketEmitOp> {
    ExpandEmitPattern(MLIRContext *context) : OpRewritePattern(context) {}
    mlir::LogicalResult matchAndRewrite(P4CoreLib::PacketEmitOp emitOp,
                                        mlir::PatternRewriter &rewriter) const override {
        static constexpr unsigned hsIdBitWidth = 32;
        auto dstPkt = emitOp.getPacketOut();
        auto emitArg = emitOp.getHdr();
        auto loc = emitOp.getLoc();
        if (auto structType = mlir::dyn_cast<P4HIR::StructType>(emitArg.getType())) {
            auto elements = structType.getFields();
            for (const auto &elt : elements) {
                auto extrData = rewriter.create<P4HIR::StructExtractOp>(loc, emitArg, elt.name);
                if (auto arrData = mlir::dyn_cast<P4HIR::HeaderStackType>(extrData.getType())) {
                    auto hsData = rewriter.create<P4HIR::StructExtractOp>(
                        loc, extrData, P4HIR::HeaderStackType::dataFieldName);
                    for (unsigned i = 0; i < arrData.getArraySize(); i++) {
                        auto idxOp = rewriter.create<P4HIR::ConstOp>(
                            loc, P4HIR::IntAttr::get(
                                     getContext(),
                                     P4HIR::BitsType::get(getContext(), hsIdBitWidth, false),
                                     llvm::APInt(hsIdBitWidth, i)));
                        auto arrItem = rewriter.create<P4HIR::ArrayGetOp>(loc, hsData, idxOp);
                        rewriter.create<P4CoreLib::PacketEmitOp>(loc, dstPkt, arrItem);
                    }
                } else {
                    rewriter.create<P4CoreLib::PacketEmitOp>(loc, dstPkt, extrData);
                }
            }
            rewriter.eraseOp(emitOp);
            return mlir::success();
        }
        return mlir::failure();
    }
};

void ExpandEmitPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandEmitPattern>(patterns.getContext());

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) signalPassFailure();
}
}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createExpandEmitPass() {
    return std::make_unique<ExpandEmitPass>();
}
