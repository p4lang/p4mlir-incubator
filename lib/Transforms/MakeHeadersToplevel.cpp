// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-make-headers-toplevel"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_MAKEHEADERSTOPLEVEL
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct MakeHeadersToplevelPass
    : public P4::P4MLIR::impl::MakeHeadersToplevelBase<MakeHeadersToplevelPass> {
    void runOnOperation() override;
};

}  // end namespace

using FieldLocation = IRUtils::LocationMapping::FieldLocation;

enum CollectionsIndices : unsigned {
    C_HEADERS = 0,
    C_METADATA = 1,
    C_COUNT = 2,
};

mlir::LogicalResult makeHeadersToplevel(mlir::FunctionOpInterface object,
                                        mlir::BlockArgument hdrsArg, mlir::BlockArgument metaArg) {
    auto *ctx = object.getContext();

    if (!mlir::isa<P4HIR::ReferenceType>(hdrsArg.getType()))
        return mlir::emitError(hdrsArg.getLoc(), "Expected reference-typed headers argument");

    if (!mlir::isa<P4HIR::ReferenceType>(metaArg.getType()))
        return mlir::emitError(metaArg.getLoc(), "Expected reference-typed metadata argument");

    auto hdrsType = P4HIR::unref<P4HIR::StructType>(hdrsArg.getType());
    auto metaType = P4HIR::unref<P4HIR::StructType>(metaArg.getType());
    IRUtils::LocationMapping mapping({hdrsType, metaType}, C_COUNT);

    // Mark all header types.
    P4HIR::FieldPath::forEachFieldPath(hdrsType, [&](P4HIR::FieldPath path) {
        if (mlir::isa<P4HIR::HeaderType, P4HIR::HeaderUnionType, P4HIR::HeaderStackType>(
                path.getType())) {
            mapping.markForMapping(FieldLocation(C_HEADERS, path), C_HEADERS);
            return mlir::WalkResult::skip();
        }

        return mlir::WalkResult::advance();
    });

    // Collect sorted header paths.
    auto hdrPaths = llvm::to_vector(mapping.getFromPaths(C_HEADERS));
    llvm::sort(hdrPaths);

    // Collect sorted metadata paths.
    auto hdrsMetaPaths =
        llvm::map_to_vector(mapping.getUnmapped(C_HEADERS), [](const auto &e) { return e.second; });
    llvm::sort(hdrsMetaPaths);

    LLVM_DEBUG({
        llvm::dbgs() << "Header paths:\n";
        for (auto path : hdrPaths) llvm::dbgs() << "  " << path << "\n";

        llvm::dbgs() << "Header Metadata paths:\n";
        for (auto path : hdrsMetaPaths) llvm::dbgs() << "  " << path << "\n";
    });

    // Build new, seperated and flattened types for headers and metadata.
    auto newHdrsType = IRUtils::createStructType(ctx, hdrsType.getName(), hdrPaths);

    auto newMetaType = metaType;
    if (!hdrsMetaPaths.empty()) {
        auto movedMetaType =
            IRUtils::createStructType(ctx, hdrsType.getName() + "_Metadata", hdrsMetaPaths);
        auto movedMetaName = mlir::StringAttr::get(ctx, "hdr_metadata$");

        auto metaFields = llvm::to_vector(metaType.getElements());
        metaFields.emplace_back(movedMetaName, movedMetaType);
        newMetaType = P4HIR::StructType::get(ctx, metaType.getName(), metaFields);
    }

    mapping.setToTypes({newHdrsType, newMetaType});

    LLVM_DEBUG({
        llvm::dbgs() << "New headers type: " << newHdrsType << "\n";
        llvm::dbgs() << "New metadata type: " << newMetaType << "\n";
    });

    // Map original header paths to the new flattened ones.
    auto newHdrPaths = llvm::to_vector(P4HIR::FieldPath(newHdrsType).children());
    mapping.addMapping(C_HEADERS, hdrPaths, C_HEADERS, newHdrPaths);

    // Map original header metadata paths to the new flattened ones.
    auto newHdrsMetaPaths =
        llvm::to_vector(P4HIR::FieldPath(newMetaType)[newMetaType.getFieldCount() - 1].children());
    mapping.addMapping(C_HEADERS, hdrsMetaPaths, C_METADATA, newHdrsMetaPaths);

    // Map original metadata.
    auto realMetaPaths = llvm::to_vector(P4HIR::FieldPath(metaType).children());
    auto newRealMetaPaths = llvm::to_vector(P4HIR::FieldPath(newMetaType).children());
    newRealMetaPaths.pop_back();
    mapping.addMapping(C_METADATA, realMetaPaths, C_METADATA, newRealMetaPaths);

    LLVM_DEBUG({
        llvm::dbgs() << "Mapping:\n";
        for (auto [from, to] : mapping.getMapping(C_HEADERS)) {
            auto desc = (to.first == C_HEADERS) ? "hdrs" : "meta";
            llvm::dbgs() << "  " << from.second << " -> " << desc << ", " << to.second << "\n";
        }
    });

    if (!mapping.verify()) return mlir::failure();

    auto maybeNewArgs = IRUtils::cloneFunctionLikeArgs(
        object, {hdrsArg, metaArg},
        {P4HIR::ReferenceType::get(newHdrsType), P4HIR::ReferenceType::get(newMetaType)});
    if (failed(maybeNewArgs)) return mlir::failure();

    auto newHdrsArg = (*maybeNewArgs)[0];
    auto newMetaArg = (*maybeNewArgs)[1];

    // Transform the arguments using the built location mapping.
    mlir::IRRewriter rewriter(ctx);
    IRUtils::LocationRewriter lr(rewriter, mapping, {hdrsArg, metaArg}, {newHdrsArg, newMetaArg});
    if (failed(lr.transform())) return mlir::failure();

    if (failed(IRUtils::eraseFunctionLikeArgs(object, {hdrsArg, metaArg}))) return mlir::failure();

    return mlir::success();
}

void MakeHeadersToplevelPass::runOnOperation() {
    auto obj = getOperation();
    if (failed(makeHeadersToplevel(obj, obj.getArguments()[hdrsArgIdx],
                                   obj.getArguments()[metaArgIdx])))
        signalPassFailure();
}

std::unique_ptr<Pass> createMakeHeadersToplevelPass() {
    return std::make_unique<MakeHeadersToplevelPass>();
}

}  // namespace P4::P4MLIR
