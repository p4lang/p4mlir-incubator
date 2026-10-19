// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "p4mlir/Transforms/LocalizeArgument.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"
#include "p4mlir/Utilities/IRUtils.h"
#include "p4mlir/Utilities/LocationRewriter.h"

#define DEBUG_TYPE "p4hir-localize-argument"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOCALIZEARGUMENT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct LocalizeArgumentPass : public P4::P4MLIR::impl::LocalizeArgumentBase<LocalizeArgumentPass> {
    void runOnOperation() override;
};

}  // end namespace

// Return accessed root fields for `arg`.
static llvm::SmallVector<P4HIR::FieldPath> analyzeFields(mlir::BlockArgument arg) {
    llvm::SmallVector<P4HIR::FieldPath> fields;
    auto leafCb = [&](mlir::Operation *op, const mlir::OpOperand &,
                      P4HIR::FieldPath path) -> mlir::WalkResult {
        if (!mlir::isa<P4HIR::ControlLocalOp>(op)) fields.push_back(path);
        return mlir::success();
    };

    IRUtils::P4HIRPathWalker::forLeafs(leafCb).walk(arg);
    P4HIR::FieldPath::computeRootFields(fields);
    return fields;
}

mlir::LogicalResult localizeArg(llvm::ArrayRef<mlir::BlockArgument> args,
                                llvm::SmallVector<P4HIR::FieldPath> escapingFields /* = {} */,
                                llvm::StringRef argNameTemplate /* = "" */) {
    if (args.empty()) return mlir::success();

    mlir::Type commonType = args.front().getType();
    auto *ctx = commonType.getContext();
    for (auto arg : args)
        if (arg.getType() != commonType)
            mlir::emitError(arg.getLoc(),
                            "Expected all arguments to have the same type for localization");

    if (!mlir::isa<P4HIR::ReferenceType>(commonType))
        mlir::emitError(args.front().getLoc(),
                        "Expected reference type arguments for localization");

    using ObjectInfo = std::tuple<mlir::FunctionOpInterface, mlir::BlockArgument,
                                  llvm::SmallVector<P4HIR::FieldPath>>;
    llvm::SmallVector<ObjectInfo> info;

    // Collect accessed root fields for all top-level objects.
    for (auto arg : args) {
        mlir::Operation *parent = arg.getOwner()->getParentOp();
        auto parentObj = mlir::dyn_cast_if_present<mlir::FunctionOpInterface>(parent);
        assert((parentObj && arg.getOwner() == &parentObj.getFunctionBody().front()) &&
               "Expected entry block argument of function-like object");

        auto fields = analyzeFields(arg);
        info.emplace_back(parentObj, arg, std::move(fields));
    }

    llvm::DenseSet<mlir::FunctionOpInterface> objSet;
    for (const auto &[obj, arg, paths] : info)
        assert(objSet.insert(obj).second && "Expected one distinct top-level object per argument");

    // Add an additional entry to prevent localization of escaping fields.
    if (!escapingFields.empty())
        info.emplace_back(mlir::FunctionOpInterface(), mlir::BlockArgument(),
                          std::move(escapingFields));

    // Build the set of root fields that need to be shared using the following rule:
    // If a field F accessed in A is also accessed in or is part of a field accessed in B, then F
    // needs to be part of the shared fields.
    llvm::SmallVector<P4HIR::FieldPath> sharedFields;

    auto unionSharedFields = [&](auto &from, auto &to) {
        for (auto path : from) {
            auto [rootField, rest] = P4HIR::FieldPath::findRootField(path, to);
            if (!rootField) continue;

            sharedFields.push_back(rootField);
        }
    };

    // The following calls the helper O(n^2) times, but we expect localization to happen between few
    // top-level objects, usually N <= 3.
    for (const auto &[obj1, arg1, paths1] : info)
        for (const auto &[obj2, arg2, paths2] : info)
            if (obj1 != obj2) unionSharedFields(paths1, paths2);

    // Calculate the final fields that need to be shared between the top-level objects.
    P4HIR::FieldPath::computeRootFields(sharedFields);

    LLVM_DEBUG({
        llvm::dbgs() << "Shared state fields:\n";
        for (auto path : sharedFields) llvm::dbgs() << "  " << path << "\n";
    });

    if (argNameTemplate.empty()) argNameTemplate = "state";

    // Build a new type for the argument replacements that only holds `sharedFields`.
    auto sharedType = IRUtils::createStructType(ctx, ("shared_" + argNameTemplate), sharedFields);

    for (auto &[obj, arg, paths] : info) {
        if (!obj) continue;

        auto maybeNewArg =
            IRUtils::duplicateFunctionLikeArgs(obj, {arg}, {P4HIR::ReferenceType::get(sharedType)});
        if (failed(maybeNewArg)) return mlir::failure();

        llvm::SmallVector<mlir::Type> mappingTypes = {sharedType};
        llvm::SmallVector<mlir::Value> mappingValues = {(*maybeNewArg)[0]};
        mlir::IRRewriter rewriter(ctx);

        llvm::SmallVector<P4HIR::FieldPath> localFields;
        for (auto &path : paths) {
            auto [rootField, rest] = P4HIR::FieldPath::findRootField(path, sharedFields);
            if (!rootField) localFields.push_back(path);
        }

        LLVM_DEBUG(llvm::dbgs() << "Local state fields:\n");

        if (!localFields.empty()) {
            // Create a local variable for fields that will be localized.
            auto localType =
                IRUtils::createStructType(ctx, ("local_" + argNameTemplate), localFields);
            mappingTypes.push_back(localType);

            rewriter.setInsertionPointToStart(&obj.getFunctionBody().front());
            auto localVar = P4HIR::VariableOp::create(rewriter, arg.getLoc(),
                                                      P4HIR::ReferenceType::get(localType),
                                                      ("local_" + argNameTemplate + "$"));
            mappingValues.push_back(localVar.getRef());

            LLVM_DEBUG({
                for (auto path : localFields) llvm::dbgs() << "  " << path << "\n";
            });
        }

        // Map fields in the original argument to ones in the replacement.
        IRUtils::LocationMapping mapping({P4HIR::unref(commonType)}, mappingTypes);
        mapping.addMapping(0, sharedFields, 0, P4HIR::FieldPath(sharedType));

        // Map fields in the original argument to the local variable.
        if (!localFields.empty())
            mapping.addMapping(0, localFields, 1, P4HIR::FieldPath(mappingTypes.back()));

        if (failed(IRUtils::P4HIRLocationRewriter(rewriter, mapping, {arg}, mappingValues)
                       .transform()))
            return mlir::failure();

        if (failed(IRUtils::eraseFunctionLikeArgs(obj, {arg}))) return mlir::failure();
    }

    return mlir::success();
}

void LocalizeArgumentPass::runOnOperation() {
    auto mod = getOperation();
    auto parsers = llvm::to_vector(mod.getOps<P4HIR::ParserOp>());
    auto controls = llvm::to_vector(mod.getOps<P4HIR::ControlOp>());
    assert((parsers.size() == 1 && controls.size() == 1) && "Expected single parser and control");
    auto args = {parsers.front().getArgument(parserArgIdx),
                 controls.front().getArgument(controlArgIdx)};
    if (failed(localizeArg(args))) signalPassFailure();
}

std::unique_ptr<Pass> createLocalizeArgumentPass() {
    return std::make_unique<LocalizeArgumentPass>();
}

}  // namespace P4::P4MLIR
