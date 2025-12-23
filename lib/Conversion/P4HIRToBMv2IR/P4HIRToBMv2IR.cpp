#include <cmath>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
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
#define GEN_PASS_DEF_P4HIRTOBMV2IR
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

static constexpr StringRef conditionalNameAttrName = "conditional_name";

static FailureOr<SymbolRefAttr> getUniqueIfOpName(P4HIR::IfOp ifOp) {
    auto name = dyn_cast<SymbolRefAttr>(ifOp->getAttr(conditionalNameAttrName));
    if (!name) return ifOp.emitError("Expected conditional name");
    return name;
}

BMv2IR::FieldInfo convertFieldInfo(P4HIR::FieldInfo p4Field) {
    return BMv2IR::FieldInfo(p4Field.name, p4Field.type);
}

StringRef getStructLikeName(P4HIR::StructLikeTypeInterface structLikeTy) {
    return llvm::TypeSwitch<P4HIR::StructLikeTypeInterface, StringRef>(structLikeTy)
        .Case([](P4HIR::HeaderType headerTy) { return headerTy.getName(); })
        .Case([](P4HIR::StructType structTy) { return structTy.getName(); })
        .Default([](P4HIR::StructLikeTypeInterface) -> StringRef {
            llvm_unreachable("Unsupported StructLike Type");
        });
}

struct P4HIRToBMv2IRTypeConverter : public mlir::TypeConverter {
    P4HIRToBMv2IRTypeConverter() {
        addConversion([&](mlir::Type t) { return t; });
        addConversion([&](P4HIR::ReferenceType ty) { return convertType(ty.getObjectType()); });
        addConversion([&](P4HIR::ValidBitType valBit) {
            return P4HIR::BitsType::get(valBit.getContext(), 1, false);
        });
        addConversion([&](P4HIR::StructLikeTypeInterface structTy) -> Type {
            SmallVector<BMv2IR::FieldInfo> newFields;
            for (auto field : structTy.getFields()) {
                // We drop the validity bit since it is basically implied in BMv2 headers
                if (isa<P4HIR::ValidBitType>(field.type)) continue;
                if (!BMv2IR::HeaderType::isAllowedFieldType(field.type)) return nullptr;
                newFields.push_back(convertFieldInfo(field));
            }
            return BMv2IR::HeaderType::get(structTy.getContext(), getStructLikeName(structTy),
                                           newFields);
        });
    }
};

struct HeaderInstanceOpConversionPattern : public OpConversionPattern<BMv2IR::HeaderInstanceOp> {
    using OpConversionPattern<BMv2IR::HeaderInstanceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BMv2IR::HeaderInstanceOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto convertedTy = getTypeConverter()->convertType(op.getHeaderType());
        rewriter.replaceOpWithNewOp<BMv2IR::HeaderInstanceOp>(op, operands.getSymName(),
                                                              convertedTy, operands.getMetadata());
        return success();
    }
};

struct SymToValConversionPattern : public OpConversionPattern<BMv2IR::SymToValueOp> {
    using OpConversionPattern<BMv2IR::SymToValueOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BMv2IR::SymToValueOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto convertedTy = getTypeConverter()->convertType(op.getType());
        rewriter.replaceOpWithNewOp<BMv2IR::SymToValueOp>(op, convertedTy, op.getDecl());
        return success();
    }
};

struct ExtractOpConversionPattern : public OpConversionPattern<P4CoreLib::PacketExtractOp> {
    using OpConversionPattern<P4CoreLib::PacketExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4CoreLib::PacketExtractOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto context = op.getContext();
        auto hdr = op.getHdr();
        auto referredTy = hdr.getType().getObjectType();
        if (!isa<P4HIR::HeaderType>(referredTy))
            return op->emitError("Only headers supported as BMv2 extract arguments");
        auto symRefOp = op.getHdr().getDefiningOp<BMv2IR::SymToValueOp>();
        auto fieldName = symRefOp.getDecl().getLeafReference();
        // TODO: support non-regular extracts
        rewriter.replaceOpWithNewOp<BMv2IR::ExtractOp>(
            op, BMv2IR::ExtractKindAttr::get(context, BMv2IR::ExtractKind::Regular),
            SymbolRefAttr::get(rewriter.getContext(), fieldName), nullptr);
        return success();
    }
};

// Converts AssignOp between headers to AssignHeadersOp
struct AssignOpToAssignHeaderPattern : public OpConversionPattern<P4HIR::AssignOp> {
    AssignOpToAssignHeaderPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::AssignOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto ctx = rewriter.getContext();
        auto src = operands.getValue();
        auto dst = operands.getRef();
        auto srcHeaderInstance = src.getDefiningOp<BMv2IR::SymToValueOp>();
        if (!srcHeaderInstance) return failure();
        auto dstHeaderInstance = dst.getDefiningOp<BMv2IR::SymToValueOp>();
        if (!dstHeaderInstance) return failure();
        rewriter.replaceOpWithNewOp<BMv2IR::AssignHeaderOp>(
            op, SymbolRefAttr::get(ctx, srcHeaderInstance.getDeclAttr().getLeafReference()),
            SymbolRefAttr::get(ctx, dstHeaderInstance.getDeclAttr().getLeafReference()));
        return success();
    }
    static constexpr unsigned benefit = 100;
};

// Converts generic P4HIR::AssignOp to BMv2IR::AssignOp. This pattern has a lower benefit
// than AssignOpToAssignHeaderPattern because we want explictly emit AssignHeaderOps when
// possible.
struct AssignOpPattern : public OpConversionPattern<P4HIR::AssignOp> {
    AssignOpPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::AssignOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto src = operands.getValue();
        auto dst = operands.getRef();
        rewriter.replaceOpWithNewOp<BMv2IR::AssignOp>(op, src, dst);
        return success();
    }
    static constexpr unsigned benefit = 1;
};

// Converts CmpOps that check for the Validity Bit to just a conversion of the validty
// field to a boolean
struct CompareValidityToD2BPattern : public OpConversionPattern<P4HIR::CmpOp> {
    CompareValidityToD2BPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::CmpOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto cmpKind = op.getKind();
        if (cmpKind != P4HIR::CmpOpKind::Eq) return failure();
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        bool isValidLhs =
            isa<P4HIR::ConstOp>(lhs.getDefiningOp()) && isa<P4HIR::ValidBitType>(lhs.getType());
        bool isValidRhs =
            isa<P4HIR::ConstOp>(rhs.getDefiningOp()) && isa<P4HIR::ValidBitType>(lhs.getType());
        if (!(isValidLhs || isValidRhs)) return failure();
        Value field = isValidLhs ? operands.getRhs() : operands.getLhs();
        rewriter.replaceOpWithNewOp<BMv2IR::DataToBoolOp>(op, field);
        return success();
    }
    static constexpr unsigned benefit = 1;
};

// Drops ReadOps since we don't have the reference type in BMv2IR
struct ReadOpConversionPattern : public OpConversionPattern<P4HIR::ReadOp> {
    using OpConversionPattern<P4HIR::ReadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ReadOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto ref = op.getRef();
        rewriter.replaceOp(op, {ref});
        return success();
    }
};

// Converts StructFieldRefOp to BMv2IR::FieldOp
struct FieldRefConversionPattern : public OpConversionPattern<P4HIR::StructFieldRefOp> {
    using OpConversionPattern<P4HIR::StructFieldRefOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::StructFieldRefOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto ctx = rewriter.getContext();
        auto resTy = getTypeConverter()->convertType(op.getResult().getType());
        auto instance = op.getInput().getDefiningOp<BMv2IR::SymToValueOp>();
        if (!instance) return failure();
        auto oldName = op.getFieldName();
        auto newName = oldName == P4HIR::HeaderType::validityBit
                           ? BMv2IR::HeaderType::validBitFieldName
                           : oldName;
        rewriter.replaceOpWithNewOp<BMv2IR::FieldOp>(
            op, resTy, SymbolRefAttr::get(ctx, instance.getDecl().getLeafReference()), newName);
        return success();
    }
};

struct ParserStateOpConversionPattern : public OpConversionPattern<P4HIR::ParserStateOp> {
    using OpConversionPattern<P4HIR::ParserStateOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(P4HIR::ParserStateOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto context = rewriter.getContext();

        SmallVector<Operation *> eraseList;

        auto newState = rewriter.create<BMv2IR::ParserStateOp>(loc, op.getSymNameAttr());
        auto &transitionBlock = newState.getTransitions().emplaceBlock();
        auto &keysBlock = newState.getTransitionKeys().emplaceBlock();
        auto terminator = op.getNextTransition();

        auto conversionOk =
            TypeSwitch<Operation *, LogicalResult>(terminator)
                .Case([&](P4HIR::ParserTransitionOp transitionOp) -> LogicalResult {
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        transitionOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        transitionOp.getStateAttr(), nullptr, nullptr);
                    eraseList.push_back(transitionOp.getOperation());
                    return success();
                })
                .Case([&](P4HIR::ParserTransitionSelectOp transitionSelectOp) {
                    for (auto operand : transitionSelectOp.getArgs()) {
                        auto transitionKey = insertTransitionKey(operand.getDefiningOp(), rewriter,
                                                                 &keysBlock, eraseList);
                        if (failed(transitionKey)) return failure();
                    }

                    for (auto selectOp : transitionSelectOp.selects()) {
                        auto transition = insertTransition(selectOp, rewriter, &transitionBlock);
                        if (failed(transition)) return failure();
                    }
                    eraseList.push_back(transitionSelectOp);
                    return success();
                })
                .Case([&](P4HIR::ParserAcceptOp acceptOp) {
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        acceptOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        nullptr, nullptr, nullptr);
                    eraseList.push_back(acceptOp.getOperation());
                    return success();
                })
                .Case([&](P4HIR::ParserRejectOp rejectOp) {
                    // TODO: p4c raises a warning "Explicit transition to reject not supported on
                    // this target"
                    //  for explicit transitions to reject
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(&transitionBlock);
                    rewriter.create<BMv2IR::TransitionOp>(
                        rejectOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Default),
                        nullptr, nullptr, nullptr);
                    eraseList.push_back(rejectOp.getOperation());
                    return success();
                })
                .Default([](Operation *) { return failure(); });

        if (failed(conversionOk)) return op->emitError("Error while processing state terminator");

        // Move the remaning parser ops to their region, they will be converted by the other
        // patterns
        newState.getParserOps().takeBody(op.getBody());
        for (auto op : eraseList) rewriter.eraseOp(op);
        rewriter.replaceOp(op, newState);

        return success();
    }

 private:
    static LogicalResult insertTransitionKey(Operation *op, ConversionPatternRewriter &rewriter,
                                             Block *block, SmallVector<Operation *> &eraseList) {
        auto loc = op->getLoc();
        if (auto lookAheadOp = dyn_cast<P4CoreLib::PacketLookAheadOp>(op)) {
            // TODO: not sure how to handle offsets
            auto offset = rewriter.getI32IntegerAttr(0);
            // TODO: can PacketLookAheadOp return something other than Bit?
            auto bitTy = cast<P4HIR::BitsType>(lookAheadOp.getResult().getType());
            auto width = rewriter.getI32IntegerAttr(bitTy.getWidth());
            ConversionPatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(block);
            eraseList.push_back(lookAheadOp);
            rewriter.create<BMv2IR::LookaheadOp>(loc, offset, width);
            return success();
        }
        return op->emitError("Unhandled transition key");
    }

    static LogicalResult insertTransition(P4HIR::ParserSelectCaseOp caseOp,
                                          ConversionPatternRewriter &rewriter, Block *block) {
        auto context = caseOp.getContext();
        auto keysets = caseOp.getSelectKeys();
        auto loc = caseOp.getLoc();

        auto constValOrError = [](Value val) -> llvm::FailureOr<TypedAttr> {
            auto constOp = val.getDefiningOp<P4HIR::ConstOp>();
            if (!constOp) return failure();

            return constOp.getValue();
        };

        for (auto entry : keysets) {
            return TypeSwitch<Operation *, LogicalResult>(entry.getDefiningOp())
                .Case([&](P4HIR::SetOp setOp) -> LogicalResult {
                    auto inputs = setOp.getInput();
                    for (auto input : inputs) {
                        auto maybeConstVal = constValOrError(input);
                        if (failed(maybeConstVal)) return setOp.emitError("Unsupported set entry");
                        ConversionPatternRewriter::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToEnd(block);

                        rewriter.create<BMv2IR::TransitionOp>(
                            loc,
                            BMv2IR::TransitionKindAttr::get(context,
                                                            BMv2IR::TransitionKind::Hexstr),
                            caseOp.getStateAttr(), maybeConstVal.value(), nullptr);
                    }
                    return success();
                })
                .Case([&](P4HIR::ConstOp constOp) {
                    if (P4HIR::isUniversalSetValue(constOp.getRes())) {
                        ConversionPatternRewriter::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToEnd(block);
                        rewriter.create<BMv2IR::TransitionOp>(
                            loc,
                            BMv2IR::TransitionKindAttr::get(context,
                                                            BMv2IR::TransitionKind::Default),
                            caseOp.getStateAttr(), nullptr, nullptr);
                        return success();
                    }
                    return failure();
                })
                .Case([&](P4HIR::MaskOp maskOp) -> LogicalResult {
                    auto maybeLhsConst = constValOrError(maskOp.getLhs());
                    auto maybeRhsConst = constValOrError(maskOp.getRhs());
                    if (failed(maybeLhsConst) || failed(maybeRhsConst))
                        return maskOp.emitError("Unhandled mask op");
                    ConversionPatternRewriter::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToEnd(block);
                    rewriter.create<BMv2IR::TransitionOp>(
                        maskOp.getLoc(),
                        BMv2IR::TransitionKindAttr::get(context, BMv2IR::TransitionKind::Hexstr),
                        caseOp.getStateAttr(), maybeLhsConst.value(), maybeRhsConst.value());
                    return success();
                })
                .Default([&](Operation *) -> LogicalResult {
                    return caseOp.emitError("Unsupported select case");
                });
        }
        return failure();
    }
};

struct ParserOpConversionPattern : public OpConversionPattern<P4HIR::ParserOp> {
    using OpConversionPattern<P4HIR::ParserOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ParserOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto firstTransition = op.getStartTransition();
        auto initState = op.getStartState();
        auto newParser =
            rewriter.create<BMv2IR::ParserOp>(loc, op.getSymNameAttr(), initState.getSymbolRef());
        rewriter.eraseOp(firstTransition);
        auto &region = newParser.getRegion();
        region.takeBody(op.getRegion());
        rewriter.replaceOp(op, newParser);
        return success();
    }
};

FailureOr<BMv2IR::V1SwitchOp> getPackageInstantiationFromParentModule(Operation *op) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp) return op->emitError("No module parent");
    // TODO: consider adding an interface to support different targets transparently
    BMv2IR::V1SwitchOp packageInstantiateOp = nullptr;
    auto walkRes = moduleOp.walk([&](BMv2IR::V1SwitchOp v1switch) {
        if (packageInstantiateOp != nullptr) return WalkResult::interrupt();
        packageInstantiateOp = v1switch;
        return WalkResult::advance();
    });
    if (walkRes.wasInterrupted())
        return op->emitError("Expected only a single package instantiation");
    return packageInstantiateOp;
}

FailureOr<bool> isTopLevelControl(P4HIR::ControlOp controlOp) {
    auto packageInstantiateOp = getPackageInstantiationFromParentModule(controlOp);
    if (failed(packageInstantiateOp)) return failure();
    auto symToCheck = controlOp.getSymName();
    return symToCheck == packageInstantiateOp->getIngress().getLeafReference() ||
           symToCheck == packageInstantiateOp->getEgress().getLeafReference();
}

FailureOr<bool> isDeparserControl(P4HIR::ControlOp controlOp) {
    auto packageInstantiateOp = getPackageInstantiationFromParentModule(controlOp);
    if (failed(packageInstantiateOp)) return failure();
    auto symToCheck = controlOp.getSymName();
    return symToCheck == packageInstantiateOp->getDeparser().getLeafReference();
}

// This pattern converts top-level controls to BMv2 patterns.
// It traverses the ControlOp, converting P4HIR tables to BMv2 tables, and IfOps to BMv2
// Conditionals. It assumes that control_apply regions have been canonicalized so that they contain
// only:
// - table_apply ops
// - extracts -> switchop
// - ifop (and the ops that compute their boolean condition)
struct PipelineConversionPattern : public OpConversionPattern<P4HIR::ControlOp> {
    using OpConversionPattern<P4HIR::ControlOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ControlOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto isTopLevel = isTopLevelControl(op);
        if (failed(isTopLevel)) return failure();
        if (!isTopLevel.value()) return failure();

        auto controlApply = cast<P4HIR::ControlApplyOp>(op.getBody().front().getTerminator());
        if (controlApply.getBody().empty()) {
            // Just replace with an empty pipeline (init table will be null in the final JSON node)
            auto pipelineOp =
                rewriter.replaceOpWithNewOp<BMv2IR::PipelineOp>(op, op.getSymName(), nullptr);
            pipelineOp.getRegion().emplaceBlock();
            return success();
        }
        {
            // Before converting tables, we insert an explicit p4hir.yield op at the end of the
            // control_apply block to simplify the next_tables section generation.
            ConversionPatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&controlApply.getBody().front());
            rewriter.create<P4HIR::YieldOp>(op.getLoc(), ValueRange{});
        }
        // Convert tables
        auto tableRes = op.walk([&](P4HIR::TableOp tableOp) {
            if (failed(convertTable(tableOp, controlApply, rewriter)))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
        if (tableRes.wasInterrupted()) return failure();

        // Convert IfOps inside control_apply to BMv2IR::ConditionalOp
        auto ifRes = controlApply.walk([&](P4HIR::IfOp ifOp) {
            if (failed(convertIfOp(ifOp, controlApply, rewriter))) return WalkResult::interrupt();
            return WalkResult::advance();
        });
        if (ifRes.wasInterrupted()) return failure();

        auto maybeInitTable = getInitTable(controlApply);
        if (failed(maybeInitTable)) return failure();
        auto pipelineOp = rewriter.create<BMv2IR::PipelineOp>(op.getLoc(), op.getSymName(),
                                                              maybeInitTable.value());
        // At this point we can erase the control_apply since all the information it carried is in
        // the next_tables sections, conditionals, and init_table.
        rewriter.eraseOp(controlApply);
        pipelineOp.getRegion().takeBody(op.getRegion());
        rewriter.replaceOp(op, pipelineOp);
        return success();
    }

 private:
    static FailureOr<SymbolRefAttr> getInitTable(P4HIR::ControlApplyOp controlApplyOp) {
        auto &block = controlApplyOp.getBody().front();
        Operation *op = &block.front();
        while (op && !isa<P4HIR::TableApplyOp, P4HIR::IfOp>(op)) op = op->getNextNode();
        if (!op) return controlApplyOp.emitError("Error retrieving initial table");
        if (auto applyOp = dyn_cast<P4HIR::TableApplyOp>(op)) return applyOp.getTable();
        auto ifOp = cast<P4HIR::IfOp>(op);
        auto maybeName = getUniqueIfOpName(ifOp);
        if (failed(maybeName)) return failure();
        return maybeName.value();
    }

    static LogicalResult convertTable(P4HIR::TableOp op, P4HIR::ControlApplyOp controlApply,
                                      ConversionPatternRewriter &rewriter) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        // Build the list of actions.
        // We assume that table_actions only contain the call to the actual control action
        SmallVector<P4HIR::TableActionOp> tableActions;
        op.walk([&](P4HIR::TableActionOp actionOp) { tableActions.push_back(actionOp); });
        SmallVector<Attribute> actionCallees;
        for (auto actionOp : tableActions) {
            auto maybeActionRef = getAction(actionOp);
            if (failed(maybeActionRef)) return actionOp.emitError("Unexpected table_action");
            actionCallees.push_back(maybeActionRef.value());
        }

        auto maybeActionTables = getActionTablePairs(op, controlApply, actionCallees);
        if (failed(maybeActionTables)) return op.emitError("Error processing next_tables node");

        // TODO: add helper
        P4HIR::TableKeyOp keyOp = nullptr;
        op.walk([&](P4HIR::TableKeyOp k) {
            assert(!keyOp && "Multiple key ops");
            keyOp = k;
        });
        auto maybeKeys = getKeys(keyOp);
        if (failed(maybeKeys)) return failure();
        // TODO: add helper
        P4HIR::TableSizeOp sizeOp = nullptr;
        op.walk([&](P4HIR::TableSizeOp size) {
            assert(!sizeOp && "Multiple size ops");
            sizeOp = size;
        });

        auto tableMatchKind = getTableMatchKind(maybeKeys.value(), rewriter);
        auto sizeAttr = dyn_cast<P4HIR::IntAttr>(sizeOp.getValue());
        auto size = sizeAttr.getValue().getSExtValue();
        auto defEntryAttr = getDefaultEntry(op);
        if (failed(defEntryAttr)) return failure();

        // TODO: implement support for indirect and indirect_ws table types
        auto tableType =
            BMv2IR::TableTypeAttr::get(rewriter.getContext(), BMv2IR::TableType::Simple);
        // TODO: add support for support_timeout
        auto supportTimeout = rewriter.getBoolAttr(false);
        rewriter.replaceOpWithNewOp<BMv2IR::TableOp>(
            op, op.getSymNameAttr(), rewriter.getArrayAttr(actionCallees),
            rewriter.getArrayAttr(maybeActionTables.value()), tableType, tableMatchKind,
            rewriter.getArrayAttr(maybeKeys.value()), supportTimeout, defEntryAttr.value(),
            rewriter.getI32IntegerAttr(size));
        return success();
    }

    static BMv2IR::TableMatchKindAttr getTableMatchKind(ArrayRef<Attribute> keys,
                                                        PatternRewriter &rewriter) {
        // From the BMv2 spec:
        // The match_type for the table needs to follow the following rules:
        // * If one match field is range, the table match_type has to be range
        // * If one match field is ternary, the table match_type has to be ternary
        // * If one match field is lpm, the table match_type is either ternary or lpm Note that it
        // is not correct to have more than one lpm match field in the same table.
        // See also p4c/backends/bmv2/common/control.h
        auto hasOneOf = [&](BMv2IR::TableMatchKind kind) {
            return llvm::any_of(keys, [&](Attribute a) {
                return cast<BMv2IR::TableKeyAttr>(a).getMatchType() == kind;
            });
        };
        bool oneRange = hasOneOf(BMv2IR::TableMatchKind::Range);
        bool oneTernary = hasOneOf(BMv2IR::TableMatchKind::Ternary);
        bool oneLPM = hasOneOf(BMv2IR::TableMatchKind::LPM);

        auto tableMatchKind = BMv2IR::TableMatchKind::Exact;
        if (oneRange) tableMatchKind = BMv2IR::TableMatchKind::Range;
        if (oneTernary) tableMatchKind = BMv2IR::TableMatchKind::Ternary;
        if (oneLPM) tableMatchKind = BMv2IR::TableMatchKind::LPM;
        return BMv2IR::TableMatchKindAttr::get(rewriter.getContext(), tableMatchKind);
    }

    static FailureOr<BMv2IR::TableDefaultEntryAttr> getDefaultEntry(P4HIR::TableOp tableOp) {
        P4HIR::TableDefaultActionOp defOp = nullptr;
        tableOp.walk([&](P4HIR::TableDefaultActionOp d) {
            assert(!defOp && "Multiple key ops");
            defOp = d;
        });
        if (!defOp) return failure();
        if (defOp.getBody().getNumArguments() != 0)
            return defOp.emitError("Support for multiple args NYI");
        auto action = getAction(defOp);
        if (failed(action)) return failure();

        bool actionConst = true;
        bool actionEntryConst = true;
        std::vector<std::string> actionData;
        return BMv2IR::TableDefaultEntryAttr::get(tableOp.getContext(), action.value(), actionConst,
                                                  actionData, actionEntryConst);
    }

    static FailureOr<BMv2IR::TableMatchKind> getBMv2TableMatchKind(
        P4HIR::MatchKindAttr matchKindAttr) {
        auto val = matchKindAttr.getValue().getValue();
        if (val == "exact") return BMv2IR::TableMatchKind::Exact;
        if (val == "lpm") return BMv2IR::TableMatchKind::LPM;
        if (val == "ternary") return BMv2IR::TableMatchKind::Ternary;
        if (val == "range") return BMv2IR::TableMatchKind::Range;
        if (val == "valid") return BMv2IR::TableMatchKind::Valid;
        return failure();
    }

    static FailureOr<BMv2IR::TableKeyAttr> getKey(P4HIR::TableKeyEntryOp matchOp) {
        auto maybeMatchKind = getBMv2TableMatchKind(matchOp.getMatchKindAttr());
        if (failed(maybeMatchKind)) return matchOp.emitError("Error converting match kind");
        // Note that this would probably be more straight forward if we applied this pattern after
        // converting to BMv2IR ops
        // TODO: implement support for other match kinds
        auto readOp = llvm::dyn_cast_or_null<P4HIR::ReadOp>(matchOp.getValue().getDefiningOp());
        if (!readOp) return matchOp.emitError("Expected ReadOp");
        auto fieldOp = dyn_cast_or_null<P4HIR::StructFieldRefOp>(readOp.getRef().getDefiningOp());
        if (!fieldOp) return matchOp.emitError("Expected StructFieldRefOp");
        auto fieldName = StringAttr::get(matchOp.getContext(), fieldOp.getFieldName());
        auto symRefOp = dyn_cast_or_null<BMv2IR::SymToValueOp>(fieldOp.getInput().getDefiningOp());
        if (!symRefOp) return matchOp.emitError("Expected SymToValueOp");
        auto header = symRefOp.getDecl();
        return BMv2IR::TableKeyAttr::get(matchOp.getContext(), maybeMatchKind.value(), header,
                                         fieldName, nullptr, nullptr);
    }

    static FailureOr<SmallVector<Attribute>> getKeys(P4HIR::TableKeyOp tableKeyOp) {
        SmallVector<Attribute> res;
        auto walkRes = tableKeyOp.walk([&](P4HIR::TableKeyEntryOp matchOp) {
            auto maybeKey = getKey(matchOp);
            if (failed(maybeKey)) return WalkResult::interrupt();
            res.push_back(maybeKey.value());
            return WalkResult::advance();
        });
        if (walkRes.wasInterrupted()) return failure();
        return res;
    }

    static FailureOr<SymbolRefAttr> getAction(Operation *op) {
        bool supported = isa<P4HIR::TableActionOp, P4HIR::TableDefaultActionOp>(op);
        assert(supported && "Unhandled op");
        Region &region = op->getRegion(0);
        if (!region.hasOneBlock()) return failure();

        Block &block = region.front();

        if (!llvm::hasSingleElement(block)) return failure();

        Operation &firstOp = block.front();

        auto callOp = dyn_cast<P4HIR::CallOp>(&firstOp);
        if (!callOp) return failure();

        if (auto callee = callOp.getCallee()) return callee;

        return failure();
    }

    // In BMv2, control flow between table_apply operations in control_apply blocks is expressed
    // by the next_table entries in the table node. So in order to fill the next_table node we need
    // to:
    // * Retrieve the table_apply operation corresponding to tableOp (TODO: can there be more than
    // one?)
    // * Look at the next operation after the table_apply:
    //   - If it's another table_apply, then the next_table node contains all entries that point to
    //   the next table (one for every action)
    //   - If it's a check on hit/miss, we need to add __HIT__ and __MISS__ entries to the table
    //   - If it's a switch, we need to add an entry for every action, with the first table in the
    //   case block as next table
    //   - If it's a yield, we check the next node of the parent op.
    static FailureOr<SmallVector<Attribute>> getActionTablePairs(P4HIR::TableOp tableOp,
                                                                 P4HIR::ControlApplyOp controlApply,
                                                                 ArrayRef<Attribute> actions) {
        P4HIR::TableApplyOp applyOp = nullptr;
        controlApply.walk([&](P4HIR::TableApplyOp applOp) {
            if (applOp.getTable().getLeafReference() == tableOp.getSymName()) {
                assert(!applyOp && "Multiple table_apply");
                applyOp = applOp;
            }
        });
        Operation *nextOp = getNextApplyOrConditionalNode(applyOp);
        if (!nextOp) return tableOp.emitError("Expected next operation");

        return getActionTablePairsForNextNode(nextOp, controlApply, actions);
    }

    static FailureOr<SmallVector<Attribute>> getActionTablePairsForNextNode(
        Operation *nextOp, P4HIR::ControlApplyOp controlApply, ArrayRef<Attribute> actions) {
        assert(nextOp && "Expected valid node");
        return llvm::TypeSwitch<Operation *, FailureOr<SmallVector<Attribute>>>(nextOp)
            .Case([&](P4HIR::YieldOp yieldOp) -> FailureOr<SmallVector<Attribute>> {
                return getActionTablePairsForYieldOp(yieldOp, controlApply, actions);
            })
            .Case([&](P4HIR::TableApplyOp nextApplyOp) {
                SmallVector<Attribute> result;
                auto nextTable = nextApplyOp.getTable();
                for (auto attr : actions) {
                    auto action = cast<SymbolRefAttr>(attr);
                    result.push_back(
                        BMv2IR::ActionTableAttr::get(nextApplyOp.getContext(), action, nextTable));
                }
                return result;
            })
            .Case([&](P4HIR::IfOp ifOp) -> FailureOr<SmallVector<Attribute>> {
                SmallVector<Attribute> result;
                auto nextTable = getUniqueIfOpName(ifOp);
                if (failed(nextTable)) return failure();
                for (auto attr : actions) {
                    auto action = cast<SymbolRefAttr>(attr);
                    result.push_back(
                        BMv2IR::ActionTableAttr::get(ifOp.getContext(), action, nextTable.value()));
                }
                return result;
            })
            .Case([&](P4HIR::StructExtractOp extractOp) -> FailureOr<SmallVector<Attribute>> {
                if (extractOp.getFieldName() != "action_run")
                    return extractOp.emitError("Only action_run field supported");
                auto switchOp = dyn_cast_or_null<P4HIR::SwitchOp>(extractOp->getNextNode());
                if (!switchOp) extractOp.emitError("Expected SwitchOp after ExtractOp");
                return getNextTablesFromSwitch(switchOp, controlApply, actions);
            })
            .Default([](Operation *op) -> FailureOr<SmallVector<Attribute>> {
                return op->emitError("Unsupported operation");
            });
    }

    static Operation *getNextApplyOrConditionalNode(Operation *op) {
        auto nextOp = op->getNextNode();
        while (
            nextOp &&
            !isa<P4HIR::TableApplyOp, P4HIR::IfOp, P4HIR::YieldOp, P4HIR::StructExtractOp>(nextOp))
            nextOp = nextOp->getNextNode();
        return nextOp;
    }

    static FailureOr<SmallVector<Attribute>> getActionTablePairsForYieldOp(
        P4HIR::YieldOp yieldOp, P4HIR::ControlApplyOp controlApply, ArrayRef<Attribute> actions) {
        if (yieldOp->getParentOp() == controlApply) {
            // This is the final table, return `null` as next table for every action
            SmallVector<Attribute> result;
            for (auto attr : actions) {
                auto action = cast<SymbolRefAttr>(attr);
                result.push_back(
                    BMv2IR::ActionTableAttr::get(yieldOp->getContext(), action, nullptr));
            }
            return result;
        }
        Operation *nextOp = nullptr;
        if (auto caseOp = yieldOp->getParentOfType<P4HIR::CaseOp>()) {
            if (!caseOp) return yieldOp.emitError("Expected CaseOp parent");
            auto switchOp = cast<P4HIR::SwitchOp>(caseOp->getParentOp());
            nextOp = getNextApplyOrConditionalNode(switchOp);
        } else if (auto ifOp = yieldOp->getParentOfType<P4HIR::IfOp>()) {
            nextOp = getNextApplyOrConditionalNode(ifOp);
        }
        return getActionTablePairsForNextNode(nextOp, controlApply, actions);
    }

    static FailureOr<SmallVector<Attribute>> getNextTablesFromSwitch(
        P4HIR::SwitchOp switchOp, P4HIR::ControlApplyOp controlApplyOp,
        ArrayRef<Attribute> actions) {
        SmallVector<Attribute> result;
        SmallPtrSet<Attribute, 5> processedActions;
        for (auto caseOp : switchOp.cases()) {
            if (caseOp.getKind() == P4HIR::CaseOpKind::Equal) {
                // This assumes that the enum field and the corresponding action have the same name
                auto vals = caseOp.getValue();
                assert(vals.size() == 1 && "More than value in equal case");
                auto enumField = dyn_cast<P4HIR::EnumFieldAttr>(vals[0]);
                if (!enumField) return caseOp.emitError("Expected EnumFieldAttr");
                auto enumVal = enumField.getField().getValue();
                auto actionIt = llvm::find_if(actions, [&](Attribute a) {
                    return cast<SymbolRefAttr>(a).getLeafReference() == enumVal;
                });
                if (actionIt == actions.end())
                    return caseOp.emitError("Enum field doesn't match any action");
                auto actionSymRefAttr = cast<SymbolRefAttr>(*actionIt);
                auto nextTable =
                    dyn_cast<P4HIR::TableApplyOp>(caseOp.getCaseRegion().front().front());
                if (!nextTable)
                    return caseOp.emitError("Expected table apply as first operation of the block");
                result.push_back(BMv2IR::ActionTableAttr::get(
                    switchOp.getContext(), actionSymRefAttr, nextTable.getTable()));
                processedActions.insert(*actionIt);
            }
        }
        // For the actions that aren't covered by explicit cases, we look at the next table from the
        // default case.
        auto defaultCase = switchOp.getDefaultCase();
        if (!defaultCase) return switchOp.emitError("Expected default case");
        auto &firstCaseOp = defaultCase.getCaseRegion().front().front();
        SmallVector<Attribute> actionsToProcess;
        for (auto &a : actions) {
            if (processedActions.contains(a)) continue;
            actionsToProcess.push_back(a);
        }
        auto maybeDefaultActionTablePairs =
            getActionTablePairsForNextNode(&firstCaseOp, controlApplyOp, actionsToProcess);
        if (failed(maybeDefaultActionTablePairs)) return failure();
        auto defaultActionTablePairs = maybeDefaultActionTablePairs.value();
        result.append(defaultActionTablePairs.begin(), defaultActionTablePairs.end());
        return result;
    }

    static LogicalResult convertIfOp(P4HIR::IfOp op, P4HIR::ControlApplyOp controlApply,
                                     ConversionPatternRewriter &rewriter) {
        auto name = getUniqueIfOpName(op);
        if (failed(name)) return op.emitError("Expected conditional name");

        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(controlApply);
        auto getNextSym = [](Region &r) -> FailureOr<SymbolRefAttr> {
            if (r.empty()) return {nullptr};
            Operation *op = &r.front().front();
            while (op) {
                if (auto tableApply = dyn_cast<P4HIR::TableApplyOp>(op))
                    return tableApply.getTable();
                if (auto ifOp = dyn_cast<P4HIR::IfOp>(op)) return getUniqueIfOpName(ifOp);
                op = op->getNextNode();
            }
            return failure();
        };
        auto maybeThenSym = getNextSym(op.getThenRegion());
        auto maybeElseSym = getNextSym(op.getElseRegion());
        if (failed(maybeThenSym) || failed(maybeElseSym))
            return op.emitError("Error while computing then/else next sym");

        auto condOp =
            rewriter.create<BMv2IR::ConditionalOp>(op.getLoc(), name.value().getLeafReference(),
                                                   maybeThenSym.value(), maybeElseSym.value());

        // Clone ops the are used to compute the IfOp condition into the BMv2IR::ConditionalOp
        // region: the corresponding JSON node has a node for the boolean expression, so we isolate
        // the ops that compute the condition here. We assume that the leaf nodes in the expression
        // are Header Instances.
        // TODO: could there be other kinds of ops as leaf? Constants?
        SmallVector<Operation *> expressionOps;
        if (failed(getIfOpConditionOps(op.getLoc(), op.getCondition(), expressionOps)))
            return op.emitError("Error retrieving expression ops");
        auto &block = condOp.getConditionRegion().emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        IRMapping mapper;
        for (auto op : llvm::reverse(expressionOps)) {
            auto clonedOp = rewriter.clone(*op, mapper);
            for (auto [origResult, clonedResult] :
                 llvm::zip(op->getResults(), clonedOp->getResults())) {
                mapper.map(origResult, clonedResult);
            }
        }
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<BMv2IR::YieldOp>(
            op.getLoc(),
            ValueRange{mapper.lookup(op.getCondition().getDefiningOp())->getResult(0)});

        return success();
    }

    static LogicalResult getIfOpConditionOps(Location loc, Value v, SmallVector<Operation *> &ops) {
        auto defOp = v.getDefiningOp();
        if (!defOp) return emitError(loc, "Expected defining operation");
        ops.push_back(defOp);
        if (isa<BMv2IR::SymToValueOp>(defOp)) return success();

        for (auto &operand : defOp->getOpOperands()) {
            if (failed(getIfOpConditionOps(loc, operand.get(), ops))) return failure();
        }
        return success();
    }
};

struct DeparserConversionPattern : public OpConversionPattern<P4HIR::ControlOp> {
    using OpConversionPattern<P4HIR::ControlOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ControlOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto isDeparser = isDeparserControl(op);
        if (failed(isDeparser)) return failure();
        if (!isDeparser.value()) return failure();
        SmallVector<Attribute> headersRef;
        auto walkRes = op.walk([&](P4CoreLib::PacketEmitOp emitOp) {
            auto readOp = emitOp.getHdr().getDefiningOp<P4HIR::ReadOp>();
            if (!readOp) {
                emitOp.emitError("Expected header to come from a ReadOp");
                return WalkResult::interrupt();
            }
            auto symRefOp = readOp.getRef().getDefiningOp<BMv2IR::SymToValueOp>();
            if (!symRefOp) {
                readOp.emitError("Expected SymToValueOp");
                return WalkResult::interrupt();
            }
            headersRef.push_back(symRefOp.getDecl());
            return WalkResult::advance();
        });
        if (walkRes.wasInterrupted()) return failure();
        rewriter.replaceOpWithNewOp<BMv2IR::DeparserOp>(op, op.getSymName(),
                                                        rewriter.getArrayAttr(headersRef));
        return success();
    }
};

static void setUniqueIfOpName(P4HIR::IfOp ifOp, P4HIR::ControlOp controlOp, unsigned id) {
    auto name = conditionalNameAttrName + std::to_string(id);
    ifOp->setAttr(
        conditionalNameAttrName,
        SymbolRefAttr::get(controlOp.getSymNameAttr(),
                           {SymbolRefAttr::get(StringAttr::get(ifOp.getContext(), name))}));
}

struct P4HIRToBMv2IRPass : public P4::P4MLIR::impl::P4HIRToBmv2IRBase<P4HIRToBMv2IRPass> {
    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp module = getOperation();

        // We need to give unique names to p4hir.if operations before converting to BMv2IR because
        // BMv2 represents control flow in control_apply blocks by having `conditional` nodes in the
        // JSON, and each has an unique name.
        unsigned conditionalId = 0;
        module.walk([&](P4HIR::IfOp ifOp) {
            auto controlApplyParent = ifOp->getParentOfType<P4HIR::ControlApplyOp>();
            if (!controlApplyParent) return WalkResult::skip();
            auto controlParent = controlApplyParent->getParentOfType<P4HIR::ControlOp>();
            if (!controlParent) return WalkResult::skip();
            setUniqueIfOpName(ifOp, controlParent, conditionalId);
            conditionalId++;
            return WalkResult::advance();
        });

        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        P4HIRToBMv2IRTypeConverter converter;
        patterns
            .add<HeaderInstanceOpConversionPattern, ParserOpConversionPattern,
                 ParserStateOpConversionPattern, ExtractOpConversionPattern,
                 AssignOpToAssignHeaderPattern, AssignOpPattern, ReadOpConversionPattern,
                 FieldRefConversionPattern, SymToValConversionPattern, CompareValidityToD2BPattern,
                 PipelineConversionPattern, DeparserConversionPattern>(converter, &context);

        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        auto isHeaderOrRef = [](Type ty) {
            if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
                return isa<BMv2IR::HeaderType>(refTy.getObjectType());
            return isa<BMv2IR::HeaderType>(ty);
        };
        target.addDynamicallyLegalOp<BMv2IR::HeaderInstanceOp>(
            [&](BMv2IR::HeaderInstanceOp headerInstanceOp) {
                return isHeaderOrRef(headerInstanceOp.getHeaderType());
            });
        target.addDynamicallyLegalOp<BMv2IR::SymToValueOp>(
            [&](BMv2IR::SymToValueOp op) { return converter.isLegal(op); });
        target.addIllegalOp<P4HIR::ParserOp>();
        target.addIllegalOp<P4HIR::ParserStateOp>();
        target.addIllegalOp<P4CoreLib::PacketExtractOp>();
        target.addIllegalOp<P4HIR::AssignOp>();
        target.addIllegalOp<P4HIR::StructFieldRefOp>();
        target.addIllegalOp<P4HIR::ReadOp>();
        target.addIllegalOp<P4HIR::CmpOp>();
        target.addDynamicallyLegalOp<P4HIR::ControlOp>([](P4HIR::ControlOp controlOp) {
            auto isTopLevel = isTopLevelControl(controlOp);
            auto isDeparser = isDeparserControl(controlOp);
            if (failed(isTopLevel) || failed(isDeparser)) return true;
            return !isTopLevel.value() && !isDeparser.value();
        });

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
        // Drop block arguments of ParserOp and PipelineOp since they should be unused after
        // conversion
        module.walk([](BMv2IR::ParserOp parserOp) {
            auto &region = parserOp.getBody();
            while (region.getNumArguments() > 0) region.eraseArgument(0);
        });
        module.walk([](BMv2IR::PipelineOp pipelineOp) {
            auto &region = pipelineOp.getBody();
            while (region.getNumArguments() > 0) region.eraseArgument(0);
        });
    }
};
}  // anonymous namespace
