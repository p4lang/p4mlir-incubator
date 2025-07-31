#include <optional>
#include <string>
#include <tuple>
#include <vector>

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

class CastSelectArguments : public mlir::OpRewritePattern<P4HIR::ParserTransitionSelectOp> {
 public:
    using OpRewritePattern<P4HIR::ParserTransitionSelectOp>::OpRewritePattern;

    virtual ~CastSelectArguments() {}

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp selectOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto selectArgs = selectOp.getArgs();
        bool hasArgAdjustments =
            llvm::any_of(selectArgs, [this](mlir::Value arg) { return shouldAdjustArg(arg); });

        if (!hasArgAdjustments)
            return rewriter.notifyMatchFailure(
                selectOp.getLoc(), "Select doesn't have applicable argument adjustments.");

        llvm::SmallVector<mlir::Value, 4> newArgs;
        llvm::SmallVector<mlir::Type, 4> newTypes;
        rewriter.setInsertionPoint(selectOp);

        // Replace arguments that we want to adjust their types with a cast.
        for (auto arg : selectArgs) {
            if (shouldAdjustArg(arg)) {
                mlir::Type castType = getAdjustedArgType(rewriter.getContext(), arg);
                mlir::Value argInt = rewriter.create<P4HIR::CastOp>(arg.getLoc(), castType, arg);
                newArgs.push_back(argInt);
                newTypes.push_back(castType);
            } else {
                newArgs.push_back(arg);
                newTypes.push_back(nullptr);
            }
        }

        rewriter.modifyOpInPlace(selectOp, [&]() { selectOp.getArgsMutable().assign(newArgs); });

        // Replace corresponding set operations or constants in select cases.
        for (auto selectCase : selectOp.selects()) {
            if (selectCase.isDefault()) continue;

            auto yield = mlir::cast<P4HIR::YieldOp>(selectCase.getTerminator());
            rewriter.setInsertionPoint(yield);

            llvm::SmallVector<mlir::Value, 4> newArgs;
            for (auto [newType, yieldArg] : llvm::zip_equal(newTypes, yield.getArgs())) {
                if (!newType || P4HIR::isUniversalSetValue(yieldArg))
                    newArgs.push_back(yieldArg);
                else
                    newArgs.push_back(castKeysetValue(rewriter, newType, yieldArg));
            }

            rewriter.modifyOpInPlace(selectOp, [&]() { yield.getArgsMutable().assign(newArgs); });
        }

        return mlir::success();
    }

 protected:
    virtual bool shouldAdjustArg(mlir::Value arg) const = 0;
    virtual mlir::Type getAdjustedArgType(mlir::MLIRContext *context, mlir::Value arg) const = 0;

 private:
    // Create appropriate casts to convert `keyset` to a set with element type `newType`.
    mlir::Value castKeysetValue(mlir::PatternRewriter &rewriter, mlir::Type newType,
                                mlir::Value keyset) const {
        auto cast = [&](mlir::Value arg) {
            return rewriter.createOrFold<P4HIR::CastOp>(arg.getLoc(), newType, arg);
        };

        if (auto setOp = keyset.getDefiningOp<P4HIR::SetOp>()) {
            auto newArg = cast(setOp.getInput().front());
            return rewriter.create<P4HIR::SetOp>(setOp.getLoc(), mlir::ValueRange(newArg));
        } else if (auto rangeOp = keyset.getDefiningOp<P4HIR::RangeOp>()) {
            auto newLhs = cast(rangeOp.getLhs());
            auto newRhs = cast(rangeOp.getRhs());
            return rewriter.create<P4HIR::RangeOp>(rangeOp.getLoc(), newLhs, newRhs);
        } else if (auto maskOp = keyset.getDefiningOp<P4HIR::MaskOp>()) {
            auto newLhs = cast(maskOp.getLhs());
            auto newRhs = cast(maskOp.getRhs());
            return rewriter.create<P4HIR::MaskOp>(maskOp.getLoc(), newLhs, newRhs);
        } else if (auto setConstOp = keyset.getDefiningOp<P4HIR::ConstOp>()) {
            auto setAttr = setConstOp.getValueAs<P4HIR::SetAttr>();

            llvm::SmallVector<mlir::Attribute> newSetAttrs;
            for (auto member : setAttr.getMembers()) {
                auto newAttr = P4HIR::foldConstantCast(newType, member);
                newSetAttrs.push_back(newAttr);
            }

            mlir::MLIRContext *ctx = rewriter.getContext();
            auto newSetAttr =
                P4HIR::SetAttr::get(P4HIR::SetType::get(ctx, newType), setAttr.getKind(),
                                    mlir::ArrayAttr::get(ctx, newSetAttrs));
            return rewriter.create<P4HIR::ConstOp>(setConstOp.getLoc(), newSetAttr);
        } else {
            llvm_unreachable("Impossible set operation.");
            return mlir::Value();
        }
    }
};

class ReplaceBoolWithInt : public CastSelectArguments {
 public:
    using CastSelectArguments::CastSelectArguments;

 protected:
    virtual bool shouldAdjustArg(mlir::Value arg) const override {
        return mlir::isa<P4HIR::BoolType>(arg.getType());
    }

    virtual mlir::Type getAdjustedArgType(mlir::MLIRContext *context,
                                          mlir::Value arg) const override {
        assert(shouldAdjustArg(arg));
        return P4HIR::BitsType::get(context, 1, false);
    }
};

class ReplaceRanges : public mlir::OpRewritePattern<P4HIR::ParserTransitionSelectOp> {
 public:
    ReplaceRanges(mlir::MLIRContext *ctx, size_t caseCountLimit)
        : OpRewritePattern(ctx), caseCountLimit(caseCountLimit) {}

    using Mask = std::pair<llvm::APInt, llvm::APInt>;
    using MasksVec = std::vector<Mask>;
    using MasksVecProduct = std::vector<MasksVec>;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp selectOp,
                                        mlir::PatternRewriter &rewriter) const override {
        std::vector<std::optional<MasksVecProduct>> masksProducts;
        llvm::SmallDenseSet<mlir::Value> signedArgsToAdjust;
        bool hasAnyTransforms = false;
        size_t totalCaseCount = 0;

        // First compute all the masks required so we can decide if to transform.
        for (auto selectCase : selectOp.selects()) {
            auto &masksProduct = masksProducts.emplace_back(std::nullopt);

            if (selectCase.isDefault()) continue;

            auto yield = mlir::cast<P4HIR::YieldOp>(selectCase.getTerminator());
            bool hasTransforms = false;

            MasksVecProduct masksProductVec;
            masksProductVec.push_back({});

            for (auto [selectArg, yieldArg] :
                 llvm::zip_equal(selectOp.getArgs(), yield.getArgs())) {
                if (auto [rangeSetAttr, isMatch] = matchConstRangeSet(yieldArg); isMatch) {
                    auto masksVec = expandRange(rangeSetAttr);

                    if (cartesianSize(masksProductVec, masksVec) > caseCountLimit) {
                        // Avoid polynomial blowup for a single case.
                        auto msg = llvm::formatv("Total cases required exceed maximum limit of {0}",
                                                 caseCountLimit);
                        return rewriter.notifyMatchFailure(selectOp.getLoc(), msg);
                    }

                    masksProductVec = cartesianAppend(masksProductVec, masksVec);
                    hasTransforms = true;

                    if (getUnderlyingBitsType(selectArg.getType()).isSigned())
                        signedArgsToAdjust.insert(selectArg);
                } else {
                    // For a non-range argument append a placeholder that will be replaced
                    // with an actual argument during transformation.
                    masksProductVec = cartesianAppend(masksProductVec, getPlaceholderMask());
                }
            }

            if (hasTransforms) {
                hasAnyTransforms = true;
                totalCaseCount += masksProductVec.size();
                masksProduct = std::move(masksProductVec);
            } else {
                totalCaseCount++;
            }
        }

        if (!hasAnyTransforms) {
            // Nothing to do.
            return mlir::failure();
        } else if (totalCaseCount > caseCountLimit) {
            auto msg = llvm::formatv("Total cases required ({0}) exceed maximum limit of {1}",
                                     totalCaseCount, caseCountLimit);
            return rewriter.notifyMatchFailure(selectOp.getLoc(), msg);
        }

        if (!signedArgsToAdjust.empty()) {
            ReplaceIntWithUInt adjustArgs(rewriter.getContext(), signedArgsToAdjust);
            auto status = adjustArgs.matchAndRewrite(selectOp, rewriter);

            if (mlir::failed(status)) return status;
        }

        // Apply transformations.
        rewriter.startOpModification(selectOp);

        size_t idx = 0;
        for (auto selectCase : llvm::make_early_inc_range(selectOp.selects())) {
            const auto &masksProduct = masksProducts[idx++];

            if (!masksProduct) continue;

            // Create a new case for each mask combination.
            for (const auto &masks : masksProduct.value()) {
                rewriter.setInsertionPoint(selectCase);
                auto newSelectCase =
                    mlir::cast<P4HIR::ParserSelectCaseOp>(rewriter.clone(*selectCase));
                auto newYield = mlir::cast<P4HIR::YieldOp>(newSelectCase.getTerminator());

                rewriter.setInsertionPoint(newYield);
                auto newArgs = materializeMasks(rewriter, masks, newYield.getArgs());

                rewriter.modifyOpInPlace(selectOp,
                                         [&]() { newYield.getArgsMutable().assign(newArgs); });
            }

            rewriter.eraseOp(selectCase);
        }

        rewriter.finalizeOpModification(selectOp);

        return mlir::success();
    }

 private:
    class ReplaceIntWithUInt : public CastSelectArguments {
     public:
        ReplaceIntWithUInt(mlir::MLIRContext *context,
                           const llvm::SmallDenseSet<mlir::Value> &signedArgsToAdjust)
            : CastSelectArguments(context), signedArgsToAdjust(signedArgsToAdjust) {}

     protected:
        virtual bool shouldAdjustArg(mlir::Value arg) const override {
            return signedArgsToAdjust.count(arg);
        }

        virtual mlir::Type getAdjustedArgType(mlir::MLIRContext *context,
                                              mlir::Value arg) const override {
            auto width = getUnderlyingBitsType(arg.getType()).getWidth();
            return P4HIR::BitsType::get(context, width, false);
        }

        const llvm::SmallDenseSet<mlir::Value> &signedArgsToAdjust;
    };

    // Debug helper to return a base-2 string for `value`.
    static std::string bitStr(const llvm::APInt &value) {
        llvm::SmallVector<char, 64> buffer;
        value.toStringUnsigned(buffer, 2);
        buffer.insert(buffer.begin(), value.getBitWidth() - buffer.size(), '0');
        return std::string(buffer.begin(), buffer.end());
    }

    static P4HIR::BitsType getUnderlyingBitsType(mlir::Type type) {
        if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(type)) return bitsType;

        if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(type))
            return getUnderlyingBitsType(serEnumType.getType());

        return {};
    }

    static Mask getPlaceholderMask() {
        return Mask{llvm::APInt(0, 0, false), llvm::APInt(0, 0, false)};
    }

    static bool isPlaceholderMask(Mask m) { return m.first.getBitWidth() == 0; }

    // Find if `val` is a SetAttr representing a constant range set and return it.
    std::pair<P4HIR::SetAttr, bool> matchConstRangeSet(mlir::Value val) const {
        if (auto constOp = val.getDefiningOp<P4HIR::ConstOp>()) {
            if (auto setAttr = mlir::dyn_cast<P4HIR::SetAttr>(constOp.getValue())) {
                if (setAttr.getKind() == P4HIR::SetKind::Range) {
                    return {setAttr, true};
                }
            }
        }

        return {nullptr, false};
    }

    size_t cartesianSize(const MasksVecProduct &vecs, const MasksVec &masks) const {
        return vecs.size() * masks.size();
    }

    MasksVecProduct cartesianAppend(const MasksVecProduct &vecs, const MasksVec &masks) const {
        MasksVecProduct newVecs;

        for (const auto &v : vecs) {
            for (const auto &mask : masks) {
                auto copy(v);
                copy.push_back(mask);
                newVecs.push_back(copy);
            }
        }

        return newVecs;
    }

    MasksVecProduct cartesianAppend(const MasksVecProduct &vecs, Mask mask) const {
        return cartesianAppend(vecs, MasksVec{mask});
    }

    // Append to `result` the masks whose union equals the unsigned range [min, max].
    void expandRangeUnsigned(MasksVec &result, llvm::APInt min, llvm::APInt max) const {
        LLVM_DEBUG(llvm::dbgs() << "Expand range [" << min << ", " << max << "]:\n");
        assert(min.ule(max) && "min u<= max");

        unsigned width = min.getBitWidth();
        unsigned activeWidth = max.getActiveBits();
        llvm::APInt range_size_remaining = max - min + 1;

        // We're decomposing the range [min, max] into power-of-2 sized chunks that can
        // be represented as a MaskOp. At each iteration we find a value `stride` (the size
        // of the current chunk) which satisfying the following criteria:
        //   - (min + stride <= max) -> (stride <= 2^floor_log2(max - min + 1))
        //      so that we don't overshoot the range.
        //   - stride <= 2^countr_zero(min)
        //      so the prefix starting at the first set bit of `min` changes.
        //   - stride is maximum, ensuring a unique and optimal solution.
        while (!range_size_remaining.isZero()) {
            unsigned strideLog2 = std::min(range_size_remaining.logBase2(), min.countr_zero());
            llvm::APInt stride = llvm::APInt::getOneBitSet(width, strideLog2);
            llvm::APInt mask = llvm::APInt::getBitsSet(width, strideLog2, width);

            LLVM_DEBUG(llvm::dbgs()
                       << "  mask " << bitStr(min.trunc(activeWidth)) << " &&& "
                       << bitStr(mask.trunc(activeWidth)) << ", stride " << stride << "\n");

            result.emplace_back(min, mask);
            range_size_remaining -= stride;
            min += stride;
        }
    }

    // Return a list masks whose union equals the range described by `rangeSetAttr`.
    MasksVec expandRange(P4HIR::SetAttr rangeSetAttr) const {
        auto [min, max] = rangeSetAttr.getRangeValues();
        unsigned width = min.getBitWidth();

        auto elmType = mlir::cast<P4HIR::SetType>(rangeSetAttr.getType()).getElementType();
        bool isSigned = mlir::cast<P4HIR::BitsType>(elmType).isSigned();
        MasksVec result;

        if (!rangeSetAttr.isEmptyRange()) {
            if (isSigned && min.slt(0) && max.sge(0)) {
                expandRangeUnsigned(result, min, llvm::APInt::getAllOnes(width));
                expandRangeUnsigned(result, llvm::APInt::getZero(width), max);
            } else {
                expandRangeUnsigned(result, min, max);
            }
        }

        return result;
    }

    // Materialize the mask set constants described by `masks` to replace some of `values`.
    std::vector<mlir::Value> materializeMasks(mlir::PatternRewriter &rewriter,
                                              const MasksVec &masks,
                                              mlir::ValueRange values) const {
        auto getMaskSetAttr = [&](const llvm::APInt &val, const llvm::APInt &mask) {
            mlir::MLIRContext *ctx = rewriter.getContext();
            auto uintType = P4HIR::BitsType::get(ctx, val.getBitWidth(), false);
            auto setType = P4HIR::SetType::get(ctx, uintType);
            auto valAttr = P4HIR::IntAttr::get(uintType, val);
            auto maskAttr = P4HIR::IntAttr::get(uintType, mask);
            return P4HIR::SetAttr::get(setType, P4HIR::SetKind::Mask,
                                       mlir::ArrayAttr::get(ctx, {valAttr, maskAttr}));
        };

        std::vector<mlir::Value> result;
        for (const auto &[mask, value] : llvm::zip_equal(masks, values)) {
            if (!isPlaceholderMask(mask)) {
                auto maskSetAttr = getMaskSetAttr(mask.first, mask.second);
                auto maskSetConstant = rewriter.create<P4HIR::ConstOp>(value.getLoc(), maskSetAttr);
                result.push_back(maskSetConstant);
            } else {
                result.push_back(value);
            }
        }

        return result;
    }

    size_t caseCountLimit;
};

}  // end namespace

void SimplifySelect::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());

    if (flattenTuples) patterns.add<FlattenTuples>(patterns.getContext());
    if (replaceBools) patterns.add<ReplaceBoolWithInt>(patterns.getContext());
    if (replaceRanges)
        patterns.add<ReplaceRanges>(patterns.getContext(), replaceRangesCaseCountLimit);

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
