// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

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
    explicit ExpandEmitPattern(MLIRContext *context) : OpRewritePattern(context) {}

    mlir::LogicalResult matchAndRewrite(P4CoreLib::PacketEmitOp emitOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto dstPkt = emitOp.getPacketOut();
        auto emitArg = emitOp.getHdr();
        auto loc = emitOp.getLoc();

        return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(emitArg.getType())
            .Case<P4HIR::HeaderType>([&](mlir::Type) {
                // Nothing to do for plain header.
                return mlir::failure();
            })
            .Case<P4HIR::HeaderUnionType, P4HIR::StructType>(
                [&](P4HIR::StructLikeTypeInterface tp) -> mlir::LogicalResult {
                    auto elements = tp.getFields();
                    for (const auto &elt : elements) {
                        auto extrData =
                            P4HIR::StructExtractOp::create(rewriter, loc, emitArg, elt.name);
                        if (!mlir::isa<P4HIR::StructLikeTypeInterface>(extrData.getType()))
                            return emitOp.emitOpError() << "Invalid type contained in emit call";
                        P4CoreLib::PacketEmitOp::create(rewriter, loc, dstPkt, extrData);
                    }
                    rewriter.eraseOp(emitOp);
                    return mlir::success();
                })
            .Case<P4HIR::HeaderStackType>([&](mlir::Type) -> mlir::LogicalResult {
                auto hsData = P4HIR::StructExtractOp::create(rewriter, loc, emitArg,
                                                             P4HIR::HeaderStackType::dataFieldName);
                auto arrayType = mlir::cast<P4HIR::ArrayType>(hsData.getType());
                static constexpr unsigned hsIdBitWidth = 32;
                auto intType = P4HIR::BitsType::get(getContext(), hsIdBitWidth, false);
                for (unsigned i = 0; i < arrayType.getSize(); i++) {
                    auto idxOp =
                        P4HIR::ConstOp::create(rewriter, loc, P4HIR::IntAttr::get(intType, i));
                    auto arrItem = P4HIR::ArrayGetOp::create(rewriter, loc, hsData, idxOp);
                    if (!mlir::isa<P4HIR::StructLikeTypeInterface>(arrItem.getType()))
                        return emitOp.emitOpError() << "Invalid type contained in emit call";
                    P4CoreLib::PacketEmitOp::create(rewriter, loc, dstPkt, arrItem);
                }
                rewriter.eraseOp(emitOp);
                return mlir::success();
            })
            .Default([&](mlir::Type) {
                return emitOp.emitOpError() << "Invalid type contained in emit call";
            });
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
