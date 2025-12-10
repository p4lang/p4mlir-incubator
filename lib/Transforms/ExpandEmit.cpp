#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
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
        auto dstPkt = emitOp.getPacketOut();
        auto emitArg = emitOp.getHdr();
        auto loc = emitOp.getLoc();
        return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(emitArg.getType())
            .Case<P4HIR::StructType>([&](P4HIR::StructType tp) {
                auto elements = tp.getFields();
                for (const auto &elt : elements) {
                    auto extrData = rewriter.create<P4HIR::StructExtractOp>(loc, emitArg, elt.name);
                    rewriter.create<P4CoreLib::PacketEmitOp>(loc, dstPkt, extrData);
                }
                rewriter.eraseOp(emitOp);
                return mlir::success();
            })
            .Case<P4HIR::HeaderStackType>([&](P4HIR::HeaderStackType tp) {
                auto hsData = rewriter.create<P4HIR::StructExtractOp>(
                    loc, emitArg, P4HIR::HeaderStackType::dataFieldName);
                rewriter.create<P4CoreLib::PacketEmitOp>(loc, dstPkt, hsData);
                rewriter.eraseOp(emitOp);
                return mlir::success();
            })
            .Case<P4HIR::ArrayType>([&](P4HIR::ArrayType tp) {
                static constexpr unsigned hsIdBitWidth = 32;
                auto intType = P4HIR::BitsType::get(getContext(), hsIdBitWidth, false);
                for (unsigned i = 0; i < tp.getSize(); i++) {
                    auto idxOp =
                        rewriter.create<P4HIR::ConstOp>(loc, P4HIR::IntAttr::get(intType, i));
                    auto arrItem = rewriter.create<P4HIR::ArrayGetOp>(loc, emitArg, idxOp);
                    rewriter.create<P4CoreLib::PacketEmitOp>(loc, dstPkt, arrItem);
                }
                rewriter.eraseOp(emitOp);
                return mlir::success();
            })
            .Default([](mlir::Type tp) { return mlir::failure(); });
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
