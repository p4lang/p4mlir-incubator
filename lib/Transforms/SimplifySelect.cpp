#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-simplify-select"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SIMPLIFYSELECT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct SimplifySelect : public impl::SimplifySelectBase<SimplifySelect> {
    void runOnOperation() override;
};

// Flatten all tuples in transition_select arguments and case keysets.
class FlattenTuples : public mlir::OpRewritePattern<P4HIR::ParserTransitionSelectOp> {
 public:
    using OpRewritePattern<P4HIR::ParserTransitionSelectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp selectOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto selectArgs = selectOp.getArgs();
        bool hasTupleArgs = llvm::any_of(
            selectArgs, [](mlir::Value v) { return mlir::isa<mlir::TupleType>(v.getType()); });

        if (!hasTupleArgs)
            return rewriter.notifyMatchFailure(selectOp.getLoc(),
                                               "Select doesn't have tuple arguments.");

        // First flatten yields in case stetamenets.
        for (auto selectCase : selectOp.selects()) {
            auto yield = mlir::cast<P4HIR::YieldOp>(selectCase.getTerminator());
            rewriter.setInsertionPoint(yield);

            llvm::SmallVector<mlir::Value, 4> newArgs;
            auto callback = [&](mlir::Value value) {
                // Re-wrap case keys with p4hir.set if needed.
                if (!mlir::isa<P4HIR::SetType>(value.getType())) {
                    value = rewriter.create<P4HIR::SetOp>(value.getLoc(), mlir::ValueRange(value));
                }

                newArgs.push_back(value);
                return true;
            };

            [[maybe_unused]] bool compatible =
                flattenValues(rewriter, selectArgs.getTypes(), yield.getArgs(), true, callback);
            assert(compatible && "The structure of yield and select args must match.");

            rewriter.modifyOpInPlace(selectOp, [&]() { yield.getArgsMutable().assign(newArgs); });
        }

        // Finally flatten the arguments of the select statement.
        llvm::SmallVector<mlir::Value, 4> newArgs;
        rewriter.setInsertionPoint(selectOp);
        flattenValues(rewriter, selectArgs.getTypes(), selectArgs, false, [&](mlir::Value value) {
            newArgs.push_back(value);
            return true;
        });

        rewriter.modifyOpInPlace(selectOp, [&]() { selectOp.getArgsMutable().assign(newArgs); });

        return mlir::success();
    }

 private:
    // Helper to flatten `keys` based on the argument types of a select statement. `shape` and
    // `keys` are lists of types and values that may be primitives (int, bool, ...), tuples
    // of such primitives or sets. Performs a DFS traversal based on the structure of `shape`
    // and report `keys`'s leaf nodes through `callback`. If `allowUnwrapSet` is true then
    // allow to look through a p4hir.set operation once. Return true iff the structure of `shape`
    // and `keys` is compatible.
    static bool flattenValues(mlir::PatternRewriter &rewriter, mlir::TypeRange shape,
                              mlir::ValueRange keys, bool allowUnwrapSet,
                              llvm::function_ref<bool(mlir::Value)> callback) {
        bool isKeysUniversalSet = false;
        llvm::SmallVector<mlir::Value, 4> keysStorage;

        if (keys.size() == 1) {
            if (P4HIR::isUniversalSetValue(keys.front())) {
                isKeysUniversalSet = true;
            } else if (auto setOp = keys.front().getDefiningOp<P4HIR::SetOp>();
                       setOp && allowUnwrapSet) {
                keys = setOp.getInput();
                allowUnwrapSet = false;
            }
        }

        // If `shape` is a tuple type then unpack it together with `keys`.
        if (shape.size() == 1) {
            if (keys.size() != 1) return false;

            if (auto tupleType1 = mlir::dyn_cast<mlir::TupleType>(shape.front())) {
                shape = tupleType1.getTypes();

                if (!isKeysUniversalSet) {
                    mlir::Value val = keys.front();
                    auto tupleType2 = mlir::dyn_cast<mlir::TupleType>(val.getType());
                    if (!tupleType2 || tupleType2.getTypes().size() != shape.size()) return false;

                    for (size_t i = 0; i < shape.size(); i++)
                        keysStorage.push_back(
                            rewriter.create<P4HIR::TupleExtractOp>(val.getLoc(), val, i));

                    keys = keysStorage;
                }
            }
        }

        if (shape.size() != keys.size() && !isKeysUniversalSet) return false;

        bool isShapePrimitive = (shape.size() == 1) && !mlir::isa<mlir::TupleType>(shape.front());

        if (isShapePrimitive) {
            return callback(keys.front());
        }

        if (isKeysUniversalSet) {
            for (auto childShape : shape)
                if (!flattenValues(rewriter, childShape, keys, allowUnwrapSet, callback))
                    return false;
        } else {
            for (auto [childShape, value] : llvm::zip_equal(shape, keys))
                if (!flattenValues(rewriter, childShape, value, allowUnwrapSet, callback))
                    return false;
        }

        return true;
    }
};

}  // end namespace

void SimplifySelect::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());

    if (flattenTuples) patterns.add<FlattenTuples>(patterns.getContext());

    // Collect operations and apply patterns.
    llvm::SmallVector<mlir::Operation *, 16> ops;
    getOperation()->walk([&](P4HIR::ParserTransitionSelectOp op) { ops.push_back(op); });

    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    auto result = applyOpPatternsGreedily(ops, std::move(patterns), config);

    if (result.failed()) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createSimplifySelectPass() {
    return std::make_unique<SimplifySelect>();
}
}  // namespace P4::P4MLIR
