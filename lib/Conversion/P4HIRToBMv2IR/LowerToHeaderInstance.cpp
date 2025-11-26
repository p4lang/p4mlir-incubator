#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "lower-to-header-instance"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERTOHEADERINSTANCE
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

P4HIR::StructType isStructOrRefToStruct(mlir::Type ty) {
    if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
        return isStructOrRefToStruct(refTy.getObjectType());
    auto structTy = dyn_cast<P4HIR::StructType>(ty);
    if (!structTy) return nullptr;
    // We assume that struct don't contain other structs
    assert(
        llvm::none_of(structTy.getFields(),
                      [](P4HIR::FieldInfo field) { return isa<P4HIR::StructType>(field.type); }) &&
        "No structs within structs");
    return structTy;
}

P4HIR::HeaderType isHeaderOrRefToHeader(mlir::Type ty) {
    if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
        return isHeaderOrRefToHeader(refTy.getObjectType());
    if (auto headerTy = dyn_cast<P4HIR::HeaderType>(ty)) return headerTy;
    return nullptr;
}

// Adds instances from a StructType, splitting the struct to create separate instances for
// the header fields, and creating a new struct containing only the bit fields if necessary
LogicalResult splitStructAndAddInstances(Value val, P4HIR::StructType structTy, Location loc,
                                         StringRef parentName, ModuleOp moduleOp,
                                         PatternRewriter &rewriter) {
    auto ctx = rewriter.getContext();
    llvm::StringMap<BMv2IR::HeaderInstanceOp> instances;
    SmallPtrSet<Operation *, 5> fieldRefs;
    SmallPtrSet<Operation *, 5> bitRefs;

    // Find the StructFieldRefOps that access the struct
    for (auto user : val.getUsers()) {
        if (auto fieldRefOp = dyn_cast<P4HIR::StructFieldRefOp>(user)) {
            auto fieldTy = structTy.getFieldType(fieldRefOp.getFieldName());
            if (isa<P4HIR::HeaderType>(fieldTy))
                fieldRefs.insert(fieldRefOp);
            else if (isa<P4HIR::BitsType, P4HIR::VarBitsType>(fieldTy))
                bitRefs.insert(fieldRefOp);
            else
                return emitError(loc, "Unsupported FieldRefOp");
        } else {
            return emitError(loc, "Unsupported struct use");
        }
    }

    // Add HeaderInstanceOps for StructFieldRefOps that reference header fields
    for (auto op : fieldRefs) {
        auto fieldRefOp = cast<P4HIR::StructFieldRefOp>(op);
        auto name = fieldRefOp.getFieldName();
        auto instance = instances.find(name);
        BMv2IR::HeaderInstanceOp instanceOp = nullptr;
        PatternRewriter::InsertionGuard guard(rewriter);
        if (instance != instances.end()) {
            instanceOp = instance->second;
        } else {
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            instanceOp = rewriter.create<BMv2IR::HeaderInstanceOp>(
                loc, rewriter.getStringAttr(parentName + "_" + name),
                P4HIR::ReferenceType::get(structTy.getFieldType(name)));
            instances.insert({name, instanceOp});
        }
        rewriter.setInsertionPointAfter(fieldRefOp);
        rewriter.replaceOpWithNewOp<BMv2IR::SymToValueOp>(
            fieldRefOp, instanceOp.getHeaderType(),
            SymbolRefAttr::get(ctx, instanceOp.getSymName()));
    }

    if (bitRefs.empty()) return success();

    // Since the struct has bit fields, we create a new type dropping the header fields, and add a
    // header instance for it

    SmallVector<P4HIR::FieldInfo> bitFields;
    for (auto field : structTy.getFields()) {
        if (isa<P4HIR::BitsType, P4HIR::VarBitsType>(field.type)) bitFields.push_back(field);
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto newTy = P4HIR::StructType::get(rewriter.getContext(), structTy.getName(), bitFields,
                                        structTy.getAnnotations());
    auto newInstance = rewriter.create<BMv2IR::HeaderInstanceOp>(
        loc, rewriter.getStringAttr(parentName), P4HIR::ReferenceType::get(newTy));
    for (auto op : bitRefs) {
        auto fieldRefOp = cast<P4HIR::StructFieldRefOp>(op);
        rewriter.setInsertionPointAfter(fieldRefOp);
        auto symToVal = rewriter.create<BMv2IR::SymToValueOp>(
            fieldRefOp.getLoc(), newInstance.getHeaderType(),
            SymbolRefAttr::get(ctx, newInstance.getSymName()));
        rewriter.replaceOpWithNewOp<P4HIR::StructFieldRefOp>(fieldRefOp, symToVal,
                                                             fieldRefOp.getFieldName());
    }

    return success();
}

LogicalResult addInstanceForHeader(Operation *op, P4HIR::HeaderType headerTy, Twine name,
                                   PatternRewriter &rewriter) {
    PatternRewriter::InsertionGuard guard(rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto instance = rewriter.create<BMv2IR::HeaderInstanceOp>(
        op->getLoc(), rewriter.getStringAttr(name), P4HIR::ReferenceType::get(headerTy));
    rewriter.setInsertionPointAfter(op);
    rewriter.replaceOpWithNewOp<BMv2IR::SymToValueOp>(
        op, instance.getHeaderType(),
        SymbolRefAttr::get(rewriter.getContext(), instance.getSymName()));

    return success();
}

LogicalResult addInstanceForHeader(BlockArgument arg, P4HIR::HeaderType headerTy, Twine name,
                                   ModuleOp moduleOp, PatternRewriter &rewriter) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto newInstance = rewriter.create<BMv2IR::HeaderInstanceOp>(
        arg.getLoc(), rewriter.getStringAttr(name), P4HIR::ReferenceType::get(headerTy));

    for (auto &use : arg.getUses()) {
        Operation *user = use.getOwner();
        rewriter.setInsertionPointToStart(user->getBlock());
        auto opIndex = use.getOperandNumber();
        auto symRef = rewriter.create<BMv2IR::SymToValueOp>(
            user->getLoc(), newInstance.getHeaderType(),
            SymbolRefAttr::get(rewriter.getContext(), newInstance.getSymName()));
        user->setOperand(opIndex, symRef.getResult());
    }

    return success();
    ;
}

struct ParserOpPattern : public OpRewritePattern<P4HIR::ParserOp> {
    using OpRewritePattern<P4HIR::ParserOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserOp parserOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto moduleOp = parserOp->getParentOfType<ModuleOp>();
        if (!moduleOp) return failure();
        SmallVector<BlockArgument> argsToProcess;
        for (auto &arg : parserOp.getArguments()) {
            auto ty = arg.getType();
            std::string parentName =
                (parserOp.getSymName() + std::to_string(arg.getArgNumber())).str();
            if (auto headerTy = isHeaderOrRefToHeader(ty)) {
                if (failed(addInstanceForHeader(arg, headerTy, parentName, moduleOp, rewriter)))
                    return parserOp->emitError("Failed to process parserOp");
            } else if (auto structTy = isStructOrRefToStruct(ty)) {
                if (failed(splitStructAndAddInstances(arg, structTy, parserOp.getLoc(), parentName,
                                                      moduleOp, rewriter)))
                    return parserOp->emitError("Failed to process parserOp");
            }
        }

        return mlir::success();
    }
};

FailureOr<StringRef> getParentName(Operation *op) {
    auto parserParent = op->getParentOfType<P4HIR::ParserOp>();
    if (!parserParent) return op->emitError("Unexpected VariableOp parent");
    return parserParent.getSymName();
}

struct VariableOpPattern : public OpRewritePattern<P4HIR::VariableOp> {
    using OpRewritePattern<P4HIR::VariableOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp variableOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto moduleOp = variableOp->getParentOfType<ModuleOp>();
        auto refTy = variableOp.getType();
        auto ty = refTy.getObjectType();
        auto maybeName = variableOp.getName();
        if (!maybeName.has_value())
            return variableOp.emitError("Unnamed variable can't be lowered to header instance");
        auto name = maybeName.value();

        auto maybeParentName = getParentName(variableOp);
        if (failed(maybeParentName)) return failure();

        auto res =
            TypeSwitch<Type, LogicalResult>(ty)
                .Case([&](P4HIR::StructType structTy) -> LogicalResult {
                    if (failed(splitStructAndAddInstances(variableOp.getResult(), structTy,
                                                          variableOp.getLoc(), name, moduleOp,
                                                          rewriter)))
                        return variableOp.emitError("Error translating variableOp");
                    return success();
                })
                .Case([&](P4HIR::HeaderType headerTy) -> LogicalResult {
                    if (failed(addInstanceForHeader(
                            variableOp, headerTy, maybeParentName.value() + "_" + name, rewriter)))
                        return variableOp.emitError("Error translating variableOp");
                    return success();
                })
                .Default([](Type ty) { return failure(); });
        return res;
    }
};

struct LowerToHeaderInstancePass
    : public P4::P4MLIR::impl::LowerToHeaderInstanceBase<LowerToHeaderInstancePass> {
    void runOnOperation() override {
        auto &context = getContext();
        RewritePatternSet patterns(&context);
        ConversionTarget target(context);
        target.addLegalDialect<BMv2IR::BMv2IRDialect>();
        target.addLegalDialect<P4HIR::P4HIRDialect>();
        target.addLegalDialect<P4CoreLib::P4CoreLibDialect>();
        target.addDynamicallyLegalOp<P4HIR::ParserOp>([](P4HIR::ParserOp parserOp) {
            auto argsTy = parserOp.getArgumentTypes();
            return !llvm::any_of(argsTy, [](mlir::Type ty) {
                return isHeaderOrRefToHeader(ty) || isStructOrRefToStruct(ty);
            });
        });
        target.addDynamicallyLegalOp<P4HIR::VariableOp>([](P4HIR::VariableOp varOp) {
            auto refTy = varOp.getType();
            auto ty = refTy.getObjectType();
            return !isa<P4HIR::HeaderType>(ty) && !isStructOrRefToStruct(ty);
        });

        // TODO: add support for controls and other ops that may lead to header instances
        patterns.add<ParserOpPattern, VariableOpPattern>(patterns.getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

}  // anonymous namespace
