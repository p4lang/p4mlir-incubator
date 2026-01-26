#include <cassert>
#include <cmath>
#include <optional>
#include <string>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
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

BMv2IR::FieldInfo convertFieldInfo(P4HIR::FieldInfo p4Field) {
    return BMv2IR::FieldInfo(p4Field.name, p4Field.type);
}

StringRef getStructLikeName(P4HIR::StructLikeTypeInterface structLikeTy) {
    return llvm::TypeSwitch<P4HIR::StructLikeTypeInterface, StringRef>(structLikeTy)
        .Case([](P4HIR::HeaderType headerTy) { return headerTy.getName(); })
        .Case([](P4HIR::StructType structTy) {
            auto oldName = structTy.getName();
            // While doing type conversion we also convert the standard_metadata_t type name to
            // standard_metadata (this is required by the Simple Switch target)
            if (oldName == BMv2IR::standardMetadataOldStructName)
                return BMv2IR::standardMetadataNewStructName;
            return structTy.getName();
        })
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

// We can't use upstream's replaceAllSymbolUses because it doesn't traverse nested regions,
// but since Header Instances are added at the top level scope, we are sure that their Symbol
// Name is unique (and reference to it will not have NestedReferences) and so travrsing nested
// regions is safe
void renameHeaderInstance(BMv2IR::HeaderInstanceOp headerInstanceOp, StringRef newName,
                          Operation *root) {
    auto oldName = headerInstanceOp.getSymName();
    auto newNameAttr = StringAttr::get(headerInstanceOp.getContext(), newName);

    // Update the symbol definition
    headerInstanceOp.setSymName(newName);

    // Walk ALL operations recursively to update references
    root->walk([&](Operation *op) {
        // Update symbol references in all attributes
        for (NamedAttribute attr : op->getAttrs()) {
            if (auto symRef = dyn_cast<SymbolRefAttr>(attr.getValue())) {
                if (symRef.getLeafReference().getValue() == oldName &&
                    symRef.getNestedReferences().empty()) {
                    op->setAttr(attr.getName(), SymbolRefAttr::get(newNameAttr));
                }
            }
        }
    });
}

struct HeaderInstanceOpConversionPattern : public OpConversionPattern<BMv2IR::HeaderInstanceOp> {
    using OpConversionPattern<BMv2IR::HeaderInstanceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BMv2IR::HeaderInstanceOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto oldName = op.getSymName();
        // Rename the standard_metadata_t Header Instance to standard_metadata (the Simple Switch
        // target seems to require this)
        if (oldName == BMv2IR::standardMetadataOldStructName) {
            auto moduleOp = op->getParentOfType<ModuleOp>();
            if (!moduleOp) return op.emitError("No module parent");
            renameHeaderInstance(op, BMv2IR::standardMetadataNewStructName, moduleOp);
        }
        auto convertedTy = getTypeConverter()->convertType(op.getHeaderType());
        rewriter.replaceOpWithNewOp<BMv2IR::HeaderInstanceOp>(op, op.getSymName(), convertedTy,
                                                              operands.getMetadata());
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

struct ExtractVLOpConversionPattern
    : public OpConversionPattern<P4CoreLib::PacketExtractVariableOp> {
    using OpConversionPattern<P4CoreLib::PacketExtractVariableOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4CoreLib::PacketExtractVariableOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto context = op.getContext();
        auto hdr = op.getHdr();
        auto referredTy = hdr.getType().getObjectType();
        if (!isa<P4HIR::HeaderType>(referredTy))
            return op->emitError("Only headers supported as BMv2 extract arguments");
        auto symRefOp = op.getHdr().getDefiningOp<BMv2IR::SymToValueOp>();
        auto fieldName = symRefOp.getDecl().getLeafReference();
        auto lengthExpr = op.getVariableFieldSizeInBits();
        // TODO: support non-regular extracts
        rewriter.replaceOpWithNewOp<BMv2IR::ExtractVLOp>(
            op, BMv2IR::ExtractKindAttr::get(context, BMv2IR::ExtractKind::Regular),
            SymbolRefAttr::get(rewriter.getContext(), fieldName), lengthExpr, nullptr);
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

// Converts AssignOp on a header validity bit to AddHeaderOp
struct AssignOpToValidityPattern : public OpConversionPattern<P4HIR::AssignOp> {
    AssignOpToValidityPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::AssignOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto src = op.getValue();
        auto dst = op.getRef();
        auto constOp = src.getDefiningOp<P4HIR::ConstOp>();
        if (!isa<P4HIR::ValidBitType>(src.getType()) || !constOp) return failure();
        auto constVal = constOp.getValueAs<P4HIR::ValidityBitAttr>();
        if (!constVal) return failure();
        auto isValid = constVal.getValue() == P4HIR::ValidityBit::Valid;

        auto fieldRef = dst.getDefiningOp<P4HIR::StructFieldRefOp>();
        if (!fieldRef || fieldRef.getFieldName() != P4HIR::HeaderType::validityBit)
            return failure();
        auto symRef = fieldRef.getInput().getDefiningOp<BMv2IR::SymToValueOp>();

        if (isValid)
            rewriter.replaceOpWithNewOp<BMv2IR::AddHeaderOp>(op, symRef.getDecl());
        else
            rewriter.replaceOpWithNewOp<BMv2IR::RemoveHeaderOp>(op, symRef.getDecl());

        return success();
    }
    static constexpr unsigned benefit = 100;
};

// Converts AssignOp from p4hir.struct ops to an header, breaking them down to assigns on individual
// fields
struct StructAssignOpPattern : public OpConversionPattern<P4HIR::AssignOp> {
    StructAssignOpPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::AssignOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto src = op.getValue();
        auto dst = op.getRef();
        auto structOp = src.getDefiningOp<P4HIR::StructOp>();
        if (!structOp) return failure();
        auto type = cast<P4HIR::StructLikeTypeInterface>(structOp.getResult().getType());
        auto values = structOp.getInput();
        for (auto [val, field] : llvm::zip(values, type.getFields())) {
            auto fieldRef = rewriter.create<P4HIR::StructFieldRefOp>(op.getLoc(), dst, field.name);
            rewriter.create<P4HIR::AssignOp>(op.getLoc(), val, fieldRef);
        }

        rewriter.eraseOp(structOp);
        rewriter.eraseOp(op);

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

// Returns true if a value is a `valid` bit
bool isValid(Value val) {
    auto defOp = val.getDefiningOp();
    if (!defOp) return false;
    return isa<P4HIR::ConstOp>(defOp) && isa<P4HIR::ValidBitType>(val.getType());
}

// Converts CmpOps that check for the Validity Bit to just a conversion of the validty
// field to a boolean
struct CompareValidityToD2BPattern : public OpConversionPattern<P4HIR::CmpOp> {
    CompareValidityToD2BPattern(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<P4HIR::CmpOp>(typeConverter, context, benefit) {}

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto cmpKind = op.getKind();
        if (cmpKind != P4HIR::CmpOpKind::Eq) return failure();
        bool isValidLhs = isValid(op.getLhs());
        bool isValidRhs = isValid(op.getRhs());
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

// Converts StructExtractOp to BMv2IR::FieldOp
struct StructExtractOpConversionPattern : public OpConversionPattern<P4HIR::StructExtractOp> {
    using OpConversionPattern<P4HIR::StructExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::StructExtractOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto ctx = rewriter.getContext();
        auto resTy = getTypeConverter()->convertType(op.getResult().getType());
        auto readOp = op.getInput().getDefiningOp<P4HIR::ReadOp>();
        if (!readOp) return op.emitError("Expected input to come from ReadOp");
        auto instance = readOp.getRef().getDefiningOp<BMv2IR::SymToValueOp>();
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
                .Default(
                    [](Operation *op) { return op->emitError("Unsupported state terminator"); });

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
        return llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](P4CoreLib::PacketLookAheadOp lookAheadOp) {
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
            })
            .Case([&](P4HIR::StructExtractOp extractOp) -> LogicalResult {
                auto readOp = extractOp.getInput().getDefiningOp<P4HIR::ReadOp>();
                if (!readOp) return extractOp.emitError("Expected input to come from ReadOp");
                auto symRef = readOp.getRef().getDefiningOp<BMv2IR::SymToValueOp>();
                if (!symRef) return extractOp.emitError("Expected input to be an Header Instance");
                ConversionPatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(block);
                rewriter.create<BMv2IR::FieldOp>(loc, extractOp.getResult().getType(),
                                                 symRef.getDecl(), extractOp.getFieldNameAttr());
                eraseList.push_back(extractOp);
                eraseList.push_back(symRef);
                eraseList.push_back(readOp);
                return success();
            })
            .Case([&](P4HIR::ReadOp readOp) -> LogicalResult {
                auto fieldRefOp = readOp.getRef().getDefiningOp<P4HIR::StructFieldRefOp>();
                if (!fieldRefOp) return readOp.emitError("Expected input to come from FieldRefOp");
                auto symRef = fieldRefOp.getInput().getDefiningOp<BMv2IR::SymToValueOp>();
                if (!symRef) return fieldRefOp.emitError("Expected input to be an Header Instance");
                ConversionPatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(block);
                rewriter.create<BMv2IR::FieldOp>(loc, readOp.getResult().getType(),
                                                 symRef.getDecl(), fieldRefOp.getFieldNameAttr());
                eraseList.push_back(readOp);
                eraseList.push_back(fieldRefOp);
                eraseList.push_back(symRef);
                return success();
            })
            .Default([](Operation *op) { return op->emitError("Unhandled transition key"); });
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
                .Case([&](P4HIR::ConstOp constOp) -> LogicalResult {
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
                    if (auto setAttr = dyn_cast<P4HIR::SetAttr>(constOp.getValue())) {
                        auto setEntries = setAttr.getMembers();
                        if (setEntries.size() != 1)
                            return constOp.emitError("Unsupported number of set entries");
                        auto intAttr = dyn_cast<P4HIR::IntAttr>(setEntries[0]);
                        if (!intAttr) return constOp.emitError("Expected IntAttr");
                        ConversionPatternRewriter::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToEnd(block);
                        rewriter.create<BMv2IR::TransitionOp>(
                            loc,
                            BMv2IR::TransitionKindAttr::get(context,
                                                            BMv2IR::TransitionKind::Hexstr),
                            caseOp.getStateAttr(), intAttr, nullptr);
                        return success();
                    }
                    return constOp.emitError("Unsupported ConstOp");
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
        auto isTopLevel = BMv2IR::isTopLevelControl(op);
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

        // Generate BMv2IR::ConditionalOp for BMv2IR::IfOps inside control_apply
        auto ifRes = controlApply.walk([&](BMv2IR::IfOp ifOp) {
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
    static SymbolRefAttr getUniqueIfOpName(BMv2IR::IfOp ifOp) {
        auto controlParent = ifOp->getParentOfType<P4HIR::ControlOp>();
        return SymbolRefAttr::get(controlParent.getSymNameAttr(),
                                  {SymbolRefAttr::get(ifOp.getContext(), ifOp.getSymNameAttr())});
    }

    static FailureOr<SymbolRefAttr> getInitTable(P4HIR::ControlApplyOp controlApplyOp) {
        auto &block = controlApplyOp.getBody().front();
        Operation *op = &block.front();
        while (op && !isa<P4HIR::TableApplyOp, BMv2IR::IfOp>(op)) op = op->getNextNode();
        if (!op) return controlApplyOp.emitError("Error retrieving initial table");
        if (auto applyOp = dyn_cast<P4HIR::TableApplyOp>(op)) return applyOp.getTable();
        auto ifOp = cast<BMv2IR::IfOp>(op);
        return getUniqueIfOpName(ifOp);
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
            actionCallees.push_back(maybeActionRef->actionRef);
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
        if (failed(maybeKeys)) return op.emitError("Unable to retrieve key op");
        // TODO: add helper
        P4HIR::TableSizeOp sizeOp = nullptr;
        op.walk([&](P4HIR::TableSizeOp size) {
            assert(!sizeOp && "Multiple size ops");
            sizeOp = size;
        });
        int tableSize = 0;
        if (!sizeOp) {
            tableSize = BMv2IR::defaultTableSize;
        } else {
            auto sizeAttr = dyn_cast<P4HIR::IntAttr>(sizeOp.getValue());
            tableSize = sizeAttr.getValue().getSExtValue();
        }

        auto constEntries = getConstantEntries(op, maybeKeys.value());
        if (failed(constEntries)) return op.emitError("Error retrieving constant entries");

        auto defEntryAttr = getDefaultEntry(op);
        auto tableMatchKind = getTableMatchKind(maybeKeys.value(), rewriter);
        if (failed(defEntryAttr)) return op.emitError("Unable to compute table match kind");

        // TODO: implement support for indirect and indirect_ws table types
        auto tableType =
            BMv2IR::TableTypeAttr::get(rewriter.getContext(), BMv2IR::TableType::Simple);
        // TODO: add support for support_timeout
        auto supportTimeout = rewriter.getBoolAttr(false);
        rewriter.replaceOpWithNewOp<BMv2IR::TableOp>(
            op, op.getSymNameAttr(), rewriter.getArrayAttr(actionCallees),
            maybeActionTables.value(), tableType, tableMatchKind,
            rewriter.getArrayAttr(maybeKeys.value()), supportTimeout, defEntryAttr.value(),
            constEntries.value().size() > 0 ? constEntries.value() : nullptr,
            rewriter.getI32IntegerAttr(tableSize));
        return success();
    }

    static FailureOr<BMv2IR::MatchKeyAttr> getTableMatchKey(Attribute attr,
                                                            P4HIR::TableEntryOp entryOp,
                                                            BMv2IR::TableKeyAttr tableKey) {
        // If the table key is ternary, we have to mask the constant with 0xFFF (with the
        // appropriate width)
        auto getExactOrTernaryFromIntAttr =
            [&](P4HIR::IntAttr intAttr) -> FailureOr<BMv2IR::MatchKeyAttr> {
            switch (tableKey.getMatchType()) {
                case BMv2IR::TableMatchKind::Ternary: {
                    auto width = tableKey.getWidth(entryOp->getParentOfType<ModuleOp>());
                    if (failed(width)) return entryOp.emitError("Error retrieving table key width");
                    auto trueMask = BMv2IR::getTrueMask(entryOp.getContext(), width.value());
                    if (failed(trueMask))
                        return entryOp.emitError("Error creating true mask for ") << width.value();
                    return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                     BMv2IR::TableMatchKind::Ternary, intAttr,
                                                     trueMask.value());
                }
                case BMv2IR::TableMatchKind::Exact: {
                    return BMv2IR::MatchKeyAttr::get(
                        entryOp.getContext(), BMv2IR::TableMatchKind::Exact, intAttr, nullptr);
                }
                case BMv2IR::TableMatchKind::LPM: {
                    auto width = tableKey.getWidth(entryOp->getParentOfType<ModuleOp>());
                    if (failed(width)) return entryOp.emitError("Error retrieving table key width");
                    return BMv2IR::MatchKeyAttr::get(
                        entryOp.getContext(), BMv2IR::TableMatchKind::LPM, intAttr,
                        IntegerAttr::get(entryOp.getContext(), APSInt::get(width.value())));
                }
                default: {
                    return entryOp.emitError("Unsupported match kind for IntAttr");
                }
            }
        };
        return llvm::TypeSwitch<Attribute, FailureOr<BMv2IR::MatchKeyAttr>>(attr)
            .Case([&](P4HIR::IntAttr intAttr) { return getExactOrTernaryFromIntAttr(intAttr); })
            .Case([&](P4HIR::UniversalSetAttr universalAttr) -> FailureOr<BMv2IR::MatchKeyAttr> {
                switch (tableKey.getMatchType()) {
                    case BMv2IR::TableMatchKind::Ternary: {
                        // p4c for this case emits a `ternary` type match key, with 0x0000, 0x0000
                        // as key and mask
                        auto width = tableKey.getWidth(entryOp->getParentOfType<ModuleOp>());
                        if (failed(width))
                            return entryOp.emitError("Error retrieving table key width");
                        auto zero = BMv2IR::getWithWidth(entryOp.getContext(), 0, width.value());
                        if (failed(zero)) return entryOp.emitError("Error creating zero attribute");
                        return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                         BMv2IR::TableMatchKind::Ternary,
                                                         zero.value(), zero.value());
                    }
                    case BMv2IR::TableMatchKind::LPM: {
                        auto width = tableKey.getWidth(entryOp->getParentOfType<ModuleOp>());
                        if (failed(width))
                            return entryOp.emitError("Error retrieving table key width");
                        auto zero = BMv2IR::getWithWidth(entryOp.getContext(), 0, width.value());
                        if (failed(zero)) return entryOp.emitError("Error creating zero attribute");
                        auto zeroAttr = IntegerAttr::get(entryOp.getContext(), APSInt::get(0));
                        return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                         BMv2IR::TableMatchKind::LPM, zero.value(),
                                                         zeroAttr);
                    }
                    default:
                        return entryOp.emitError("Unsupported key type for UniversalSet");
                }
            })
            .Case([&](P4HIR::SetAttr setAttr) -> FailureOr<BMv2IR::MatchKeyAttr> {
                auto kind = setAttr.getKind();
                SmallVector<Attribute> bmv2Entries;
                switch (kind) {
                    case P4HIR::SetKind::Range: {
                        auto members = setAttr.getMembers();
                        if (members.size() != 2)
                            return entryOp.emitError("Expected two set members");
                        return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                         BMv2IR::TableMatchKind::Range, members[0],
                                                         members[1]);
                    }
                    case P4HIR::SetKind::Mask: {
                        auto members = setAttr.getMembers();
                        if (members.size() != 2)
                            return entryOp.emitError("Expected two set members");
                        switch (tableKey.getMatchType()) {
                            case BMv2IR::TableMatchKind::Ternary: {
                                return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                                 BMv2IR::TableMatchKind::Ternary,
                                                                 members[0], members[1]);
                            }
                            case BMv2IR::TableMatchKind::LPM: {
                                // The prefix_length calculation is takes from
                                // p4c/backends/bmv2/common/control.h
                                auto mask = dyn_cast<P4HIR::IntAttr>(members[1]);
                                if (!mask) return entryOp.emitError("Expected IntAttr for mask");
                                auto maskVal = mask.getUInt();
                                auto trailingZeros = llvm::countr_zero(maskVal);
                                auto width =
                                    tableKey.getWidth(entryOp->getParentOfType<ModuleOp>());
                                if (failed(width))
                                    return entryOp.emitError("Error retrieving table key width");
                                auto prefixLen = width.value() - trailingZeros;
                                auto prefixLenAttr =
                                    IntegerAttr::get(entryOp.getContext(), APSInt::get(prefixLen));
                                return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                                 BMv2IR::TableMatchKind::LPM,
                                                                 members[0], prefixLenAttr);
                            }
                            default: {
                                return entryOp.emitError("Unsupported Mask MatchType");
                            }
                        }
                    }
                    case P4HIR::SetKind::Constant: {
                        auto members = setAttr.getMembers();
                        if (members.size() != 1)
                            return entryOp.emitError("Expected one set member");
                        auto intAttr = dyn_cast<P4HIR::IntAttr>(members[0]);
                        if (!intAttr) return entryOp.emitError("Expected IntAttr");
                        return getExactOrTernaryFromIntAttr(intAttr);
                    }
                    default: {
                        return entryOp.emitError("Unsupported SetAttr kind");
                    }
                }
            })
            .Case([&](P4HIR::BoolAttr boolAttr) -> FailureOr<BMv2IR::MatchKeyAttr> {
                auto val = boolAttr.getValue();
                auto type = P4HIR::BitsType::get(entryOp.getContext(), 1, false);
                auto intAttr = P4HIR::IntAttr::get(type, val);
                if (tableKey.getMatchType() != BMv2IR::TableMatchKind::Exact)
                    return entryOp.emitError("Expected exact match on BoolAttr");
                return BMv2IR::MatchKeyAttr::get(entryOp.getContext(),
                                                 BMv2IR::TableMatchKind::Exact, intAttr, nullptr);
            })
            .Default([&](Attribute attr) {
                return entryOp.emitError("Unsupported match key attribute ")
                       << attr.getAbstractAttribute().getName();
            });
    }

    // Returns a SmallVector of BMv2IR::MatchKeyAttr for SetAttr of `product` kind or AggAttr
    static FailureOr<SmallVector<BMv2IR::MatchKeyAttr>> getTableMatchKeyForCollection(
        TypedAttr attr, P4HIR::TableEntryOp entryOp, ArrayRef<Attribute> tableKeys) {
        auto members =
            llvm::TypeSwitch<TypedAttr, FailureOr<ArrayAttr>>(attr)
                .Case([&](P4HIR::SetAttr setAttr) -> FailureOr<ArrayAttr> {
                    if (setAttr.getKind() != P4HIR::SetKind::Product)
                        return entryOp.emitError("Expected product set attribute");
                    return setAttr.getMembers();
                })
                .Case([&](P4HIR::AggAttr aggAttr) { return aggAttr.getFields(); })
                .Default([&](TypedAttr) { return entryOp.emitError("Unsupported TypedAttr"); });

        if (failed(members)) return failure();
        SmallVector<BMv2IR::MatchKeyAttr> bmv2Entries;
        if (tableKeys.size() != members->size())
            return entryOp.emitError("Expected same number of keys");
        for (auto [constKeyAttr, tableKeyAttr] : llvm::zip(members.value(), tableKeys)) {
            auto matchKey =
                getTableMatchKey(constKeyAttr, entryOp, cast<BMv2IR::TableKeyAttr>(tableKeyAttr));
            if (failed(matchKey)) return failure();
            bmv2Entries.push_back(matchKey.value());
        }
        return bmv2Entries;
    }

    static FailureOr<ArrayAttr> getConstantEntries(P4HIR::TableOp tableOp,
                                                   ArrayRef<Attribute> keys) {
        SmallVector<Attribute> bmv2Entries;
        auto walkRes = tableOp.walk([&](P4HIR::TableEntryOp entryOp) {
            auto matchKey = getTableMatchKeyForCollection(entryOp.getKeys(), entryOp, keys);
            if (failed(matchKey)) {
                entryOp.emitError("Error processing match key");
                return WalkResult::interrupt();
            }
            auto action = getAction(entryOp);
            if (failed(action)) {
                entryOp.emitError("Error processing table action");
                return WalkResult::interrupt();
            }
            bmv2Entries.push_back(BMv2IR::TableEntryAttr::get(
                entryOp.getContext(), matchKey.value(), action->actionRef, action->constArgs));
            return WalkResult::advance();
        });

        if (walkRes.wasInterrupted()) return failure();
        return ArrayAttr::get(tableOp.getContext(), bmv2Entries);
    }

    static BMv2IR::TableMatchKindAttr getTableMatchKind(ArrayRef<Attribute> keys,
                                                        PatternRewriter &rewriter) {
        // From the BMv2 spec:
        // The match_type for the table needs to follow the following rules:
        // * If one match field is range, the table match_type has to be range
        // * If one match field is ternary, the table match_type has to be ternary
        // * If one match field is lpm, the table match_type is either ternary or lpm Note that
        // it is not correct to have more than one lpm match field in the same table. See also
        // p4c/backends/bmv2/common/control.h
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
        return BMv2IR::TableDefaultEntryAttr::get(tableOp.getContext(), action->actionRef,
                                                  actionConst, actionData, actionEntryConst);
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
        // Note that this would probably be more straight forward if we applied this pattern
        // after converting to BMv2IR ops
        // TODO: implement support for other match kinds
        auto defOp = matchOp.getValue().getDefiningOp();

        if (!defOp) return matchOp.emitError("Expected defining operation for TableKeyEntryOp");
        return llvm::TypeSwitch<Operation *, FailureOr<BMv2IR::TableKeyAttr>>(defOp)
            .Case([&](P4HIR::ReadOp readOp) -> FailureOr<BMv2IR::TableKeyAttr> {
                auto fieldOp =
                    dyn_cast_or_null<P4HIR::StructFieldRefOp>(readOp.getRef().getDefiningOp());
                if (!fieldOp) return matchOp.emitError("Expected StructFieldRefOp");
                auto fieldName = StringAttr::get(matchOp.getContext(), fieldOp.getFieldName());
                auto symRefOp =
                    dyn_cast_or_null<BMv2IR::SymToValueOp>(fieldOp.getInput().getDefiningOp());
                if (!symRefOp) return matchOp.emitError("Expected SymToValueOp");
                auto header = symRefOp.getDecl();
                return BMv2IR::TableKeyAttr::get(matchOp.getContext(), maybeMatchKind.value(),
                                                 header, fieldName, nullptr,
                                                 BMv2IR::getControlPlaneName(matchOp));
            })
            .Case([&](P4HIR::StructExtractOp extractOp) -> FailureOr<BMv2IR::TableKeyAttr> {
                auto readOp = dyn_cast_or_null<P4HIR::ReadOp>(extractOp.getInput().getDefiningOp());
                if (!readOp) return extractOp.emitError("Expected ReadOp as extract input");
                auto symRefOp = dyn_cast<BMv2IR::SymToValueOp>(readOp.getRef().getDefiningOp());
                if (!symRefOp) return matchOp.emitError("Expected SymToValueOp");
                auto fieldName = StringAttr::get(matchOp.getContext(), extractOp.getFieldName());
                auto header = symRefOp.getDecl();
                return BMv2IR::TableKeyAttr::get(matchOp.getContext(), maybeMatchKind.value(),
                                                 header, fieldName, nullptr,
                                                 BMv2IR::getControlPlaneName(matchOp));
            })
            .Case([&](P4HIR::CmpOp cmpOp) -> FailureOr<BMv2IR::TableKeyAttr> {
                // p4c seems to emit `exact` match kind keys even if technically it should support
                // `valid`, so we emit `exact` matches to the `$valid$` field ref.
                if (maybeMatchKind.value() != BMv2IR::TableMatchKind::Exact)
                    return matchOp.emitError("Unexpected match kind");
                auto lhs = cmpOp.getLhs();
                auto rhs = cmpOp.getRhs();
                if (!isValid(lhs) && !isValid(rhs))
                    return matchOp.emitError("Expected validity check");
                auto validVal = isValid(lhs) ? lhs : rhs;
                auto validAttr =
                    validVal.getDefiningOp<P4HIR::ConstOp>().getValueAs<P4HIR::ValidityBitAttr>();
                if (!validAttr) return matchOp.emitError("Expected validity bit attr");
                if (validAttr.getValue() != P4HIR::ValidityBit::Valid)
                    return matchOp.emitError("Expected check for valid");
                auto fieldVal = isValid(lhs) ? rhs : lhs;
                auto readOp = fieldVal.getDefiningOp<P4HIR::ReadOp>();
                if (!readOp) return matchOp.emitError("Expected ReadOp");
                auto fieldRef = readOp.getRef().getDefiningOp<P4HIR::StructFieldRefOp>();
                if (!fieldRef) return matchOp.emitError("Expected StructFieldRefOp");
                auto symRefOp = fieldRef.getInput().getDefiningOp<BMv2IR::SymToValueOp>();
                if (!symRefOp) return matchOp.emitError("Expected SymToValueOp");
                auto header = symRefOp.getDecl();
                auto fieldName = fieldRef.getFieldName();
                if (fieldName != P4HIR::HeaderType::validityBit)
                    return matchOp.emitError("Expected match on validity bit field");
                auto newFieldName =
                    StringAttr::get(matchOp.getContext(), BMv2IR::HeaderType::validBitFieldName);
                return BMv2IR::TableKeyAttr::get(matchOp.getContext(), maybeMatchKind.value(),
                                                 header, newFieldName, nullptr,
                                                 BMv2IR::getControlPlaneName(matchOp));
            })
            .Default([](Operation *op) {
                return op->emitError("Unhandled operation when handling key entry");
            });
    }

    static FailureOr<SmallVector<Attribute>> getKeys(P4HIR::TableKeyOp tableKeyOp) {
        if (!tableKeyOp) return {{}};
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

    // Represents the call operation for TableActionOp, TableDefaultActionOp and TableEntryOp, with
    // the SymbolRef to the actual control action and the optional constant arguments
    struct TableActionCall {
        SymbolRefAttr actionRef;
        SmallVector<Attribute> constArgs;
    };

    static FailureOr<TableActionCall> getAction(Operation *op) {
        bool supported =
            isa<P4HIR::TableActionOp, P4HIR::TableDefaultActionOp, P4HIR::TableEntryOp>(op);
        assert(supported && "Unhandled op");
        Region &region = op->getRegion(0);
        if (!region.hasOneBlock()) return op->emitError("Expected region with one block");

        Block &block = region.front();

        if (!llvm::hasSingleElement(block))
            return op->emitError("Expected region with one element");

        Operation &firstOp = block.front();

        auto callOp = dyn_cast<P4HIR::CallOp>(&firstOp);
        if (!callOp) return callOp.emitError("Expected call operation");

        SmallVector<Attribute> constArgs;
        for (auto operand : callOp.getArgOperands()) {
            // BlockArgs are handled by using HeaderInstances in the corresponding control actions
            if (isa<BlockArgument>(operand)) continue;
            auto defOp = operand.getDefiningOp<P4HIR::ConstOp>();
            if (!defOp) return callOp.emitError("Unsupported call operand");
            auto val = dyn_cast<P4HIR::IntAttr>(defOp.getValue());
            if (!val) return callOp.emitError("Unsupported const value");
            constArgs.push_back(val);
        }

        return {{callOp.getCallee(), std::move(constArgs)}};
    }

    // In BMv2, control flow between table_apply operations in control_apply blocks is expressed
    // by the next_table entries in the table node. So in order to fill the next_table node we
    // need to:
    // * Retrieve the table_apply operation corresponding to tableOp (TODO: can there be more
    // than one?)
    // * Look at the next operation after the table_apply:
    //   - If it's another table_apply, then the next_table node contains all entries that point
    //   to the next table (one for every action)
    //   - If it's a check on hit/miss, we need to add __HIT__ and __MISS__ entries to the table
    //   - If it's a switch, we need to add an entry for every action, with the first table in
    //   the case block as next table
    //   - If it's a yield, we check the next node of the parent op.
    static FailureOr<Attribute> getActionTablePairs(P4HIR::TableOp tableOp,
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
        // We explcitly add a p4hir.yield op at the end of the control_apply block
        // to ensure that the block is yield terminated, so nextOp should always be non null
        if (!nextOp) return tableOp.emitError("Expected next operation");

        return getActionTablePairsForNextNode(nextOp, controlApply, actions);
    }

    static FailureOr<Attribute> getActionTablePairsForNextNode(Operation *nextOp,
                                                               P4HIR::ControlApplyOp controlApply,
                                                               ArrayRef<Attribute> actions) {
        assert(nextOp && "Expected valid node");
        return llvm::TypeSwitch<Operation *, FailureOr<Attribute>>(nextOp)
            .Case([&](P4HIR::YieldOp yieldOp) -> FailureOr<Attribute> {
                return getActionTablePairsForYieldOp(yieldOp, controlApply, actions);
            })
            .Case([&](BMv2IR::YieldOp yieldOp) -> FailureOr<Attribute> {
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
                return ArrayAttr::get(nextApplyOp.getContext(), result);
            })
            .Case([&](BMv2IR::IfOp ifOp) -> FailureOr<Attribute> {
                SmallVector<Attribute> result;
                auto nextTable = getUniqueIfOpName(ifOp);
                for (auto attr : actions) {
                    auto action = cast<SymbolRefAttr>(attr);
                    result.push_back(
                        BMv2IR::ActionTableAttr::get(ifOp.getContext(), action, nextTable));
                }
                return ArrayAttr::get(ifOp.getContext(), result);
            })
            .Case([&](P4HIR::StructExtractOp extractOp) -> FailureOr<Attribute> {
                auto fieldName = extractOp.getFieldName();
                if (fieldName == "action_run") {
                    auto switchOp = dyn_cast_or_null<P4HIR::SwitchOp>(extractOp->getNextNode());
                    if (!switchOp)
                        extractOp.emitError("Expected SwitchOp after action_run ExtractOp");
                    return getNextTablesFromSwitch(switchOp, controlApply, actions);
                } else if (fieldName == "hit" || fieldName == "miss") {
                    auto ifOp = dyn_cast_or_null<P4HIR::IfOp>(extractOp->getNextNode());
                    if (!ifOp) extractOp.emitError("Expected IfOp after hit/miss ExtractOp");
                    bool isHit = fieldName == "hit";
                    return getNextTablesFromHitMissIf(ifOp, controlApply, isHit);
                } else {
                    return extractOp.emitError("Only action_run field supported");
                }
            })
            .Default([](Operation *op) -> FailureOr<Attribute> {
                return op->emitError("Unsupported operation");
            });
    }

    static Operation *getNextApplyOrConditionalNode(Operation *op) {
        auto nextOp = op->getNextNode();
        while (nextOp && !isa<P4HIR::TableApplyOp, BMv2IR::IfOp, P4HIR::YieldOp, BMv2IR::YieldOp,
                              P4HIR::StructExtractOp>(nextOp)) {
            nextOp = nextOp->getNextNode();
        }
        return nextOp;
    }

    // Returns the next node from a terminating yield op. If the returned Operation* is null it
    // means that the yield op is the one terminating the control apply block
    static FailureOr<Operation *> getNextApplyOrConditionalForYield(
        Operation *yieldOp, P4HIR::ControlApplyOp controlApply) {
        bool isYield = isa<P4HIR::YieldOp, BMv2IR::YieldOp>(yieldOp);
        assert(isYield && "Expected YieldOp");
        if (yieldOp->getParentOp() == controlApply) return {nullptr};
        auto nextOp = llvm::TypeSwitch<Operation *, FailureOr<Operation *>>(yieldOp->getParentOp())
                          .Case([&](P4HIR::CaseOp caseOp) -> FailureOr<Operation *> {
                              if (!caseOp) return yieldOp->emitError("Expected CaseOp parent");
                              auto switchOp = cast<P4HIR::SwitchOp>(caseOp->getParentOp());
                              return getNextApplyOrConditionalNode(switchOp);
                          })
                          .Case<BMv2IR::IfOp, P4HIR::IfOp>(
                              [](Operation *ifOp) { return getNextApplyOrConditionalNode(ifOp); })
                          .Default([&](Operation *op) -> FailureOr<Operation *> {
                              return yieldOp->emitError("Unhandled yield parent");
                          });
        if (failed(nextOp)) return failure();
        return nextOp;
    }

    static FailureOr<Attribute> getActionTablePairsForYieldOp(Operation *yieldOp,
                                                              P4HIR::ControlApplyOp controlApply,
                                                              ArrayRef<Attribute> actions) {
        bool isYield = isa<P4HIR::YieldOp, BMv2IR::YieldOp>(yieldOp);
        assert(isYield && "Expected YieldOp");
        auto nextOp = getNextApplyOrConditionalForYield(yieldOp, controlApply);
        if (failed(nextOp)) return failure();
        if (nextOp.value() == nullptr) {
            // This is the final table, return `null` as next table for every action
            SmallVector<Attribute> result;
            for (auto attr : actions) {
                auto action = cast<SymbolRefAttr>(attr);
                result.push_back(
                    BMv2IR::ActionTableAttr::get(yieldOp->getContext(), action, nullptr));
            }
            return ArrayAttr::get(yieldOp->getContext(), result);
        }
        return getActionTablePairsForNextNode(nextOp.value(), controlApply, actions);
    }

    static FailureOr<Attribute> getNextTablesFromHitMissIf(P4HIR::IfOp ifOp,
                                                           P4HIR::ControlApplyOp controlApplyOp,
                                                           bool conditionIsHit) {
        // FIXME: support conditionals
        auto thenApply = dyn_cast<P4HIR::TableApplyOp>(ifOp.getThenRegion().front().front());
        if (!thenApply) return ifOp.emitError("Expected table apply in then region");
        P4HIR::TableApplyOp elseApply;
        if (!ifOp.getElseRegion().empty()) {
            elseApply = dyn_cast<P4HIR::TableApplyOp>(ifOp.getElseRegion().front().front());
            if (!elseApply) return ifOp.emitError("Expected table apply in else region");
        } else {
            auto nextNode = getNextApplyOrConditionalNode(ifOp);
            auto nextTable = dyn_cast<P4HIR::TableApplyOp>(nextNode);
            if (!nextTable) return ifOp.emitError("Expected table apply in next node after IfOp");
            elseApply = nextTable;
        }
        SymbolRefAttr hitRef = conditionIsHit ? thenApply.getTableAttr() : elseApply.getTableAttr();
        SymbolRefAttr missRef =
            conditionIsHit ? elseApply.getTableAttr() : thenApply.getTableAttr();
        return BMv2IR::HitOrMissAttr::get(ifOp.getContext(), hitRef, missRef);
    }

    static FailureOr<Attribute> getNextTablesFromSwitch(P4HIR::SwitchOp switchOp,
                                                        P4HIR::ControlApplyOp controlApplyOp,
                                                        ArrayRef<Attribute> actions) {
        SmallVector<Attribute> result;
        SmallPtrSet<Attribute, 5> processedActions;
        for (auto caseOp : switchOp.cases()) {
            if (caseOp.getKind() == P4HIR::CaseOpKind::Equal) {
                // This assumes that the enum field and the corresponding action have the same
                // name
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
                // FIXME: this should also support conditionals
                auto nextTable =
                    dyn_cast<P4HIR::TableApplyOp>(caseOp.getCaseRegion().front().front());
                if (!nextTable)
                    return caseOp.emitError("Expected table apply as first operation of the block");
                result.push_back(BMv2IR::ActionTableAttr::get(
                    switchOp.getContext(), actionSymRefAttr, nextTable.getTable()));
                processedActions.insert(*actionIt);
            }
        }
        // For the actions that aren't covered by explicit cases, we look at the next table from
        // the default case.
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
        auto defaultActionTablePairs = cast<ArrayAttr>(maybeDefaultActionTablePairs.value());
        result.append(defaultActionTablePairs.begin(), defaultActionTablePairs.end());
        return ArrayAttr::get(switchOp.getContext(), result);
    }

    // FIXME: remove control_apply arg
    static FailureOr<SymbolRefAttr> getSymRef(Operation *op, P4HIR::ControlApplyOp controlApply) {
        if (auto tableApply = dyn_cast<P4HIR::TableApplyOp>(op)) return tableApply.getTable();
        if (auto ifOp = dyn_cast<BMv2IR::IfOp>(op)) {
            return getUniqueIfOpName(ifOp);
        }
        return failure();
    }

    static LogicalResult convertIfOp(BMv2IR::IfOp op, P4HIR::ControlApplyOp controlApply,
                                     ConversionPatternRewriter &rewriter) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(controlApply);
        auto getNextSym = [&controlApply, &op,
                           ctx = op.getContext()](Region &r) -> FailureOr<SymbolRefAttr> {
            // First we check for empty regions by looking at the next node after the if
            if (r.empty() || r.front().getOperations().size() == 1) {
                auto nextNode = getNextApplyOrConditionalNode(op);
                if (isa<P4HIR::YieldOp, BMv2IR::YieldOp>(nextNode)) {
                    auto yieldNextNode = getNextApplyOrConditionalForYield(nextNode, controlApply);
                    if (failed(yieldNextNode)) return failure();
                    if (yieldNextNode.value() == nullptr) return {nullptr};
                    nextNode = yieldNextNode.value();
                }
                auto symRef = getSymRef(nextNode, controlApply);
                if (failed(symRef)) return failure();
                return symRef.value();
            }
            // At this point we have already checked for an empty region (or a region with just
            // a yield) Any other op should be either a table_apply or an if op
            Operation *regionOp = &r.front().front();
            while (regionOp) {
                if (auto tableApply = dyn_cast<P4HIR::TableApplyOp>(regionOp))
                    return tableApply.getTable();
                if (auto ifOp = dyn_cast<BMv2IR::IfOp>(regionOp)) {
                    return getUniqueIfOpName(ifOp);
                }
                regionOp = regionOp->getNextNode();
            }
            return op.emitError("Unexpected region");
        };
        auto maybeThenSym = getNextSym(op.getThenRegion());
        auto maybeElseSym = getNextSym(op.getElseRegion());
        if (failed(maybeThenSym)) return op.emitError("Error while computing then next sym");
        if (failed(maybeElseSym)) return op.emitError("Error while computing else next sym");

        auto condOp = rewriter.create<BMv2IR::ConditionalOp>(
            op.getLoc(), op.getSymName(), maybeThenSym.value(), maybeElseSym.value());
        condOp.getConditionRegion().takeBody(op.getConditionRegion());

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
        auto isDeparser = BMv2IR::isDeparserControl(op);
        if (failed(isDeparser)) return failure();
        if (!isDeparser.value()) return failure();
        SmallVector<Attribute> headersRef;
        auto walkRes = op.walk([&](P4CoreLib::PacketEmitOp emitOp) {
            auto defOp = emitOp.getHdr().getDefiningOp();
            if (!defOp) {
                emitOp.emitError("Expected defining op for emit header");
                return WalkResult::interrupt();
            }
            auto ref = llvm::TypeSwitch<Operation *, FailureOr<SymbolRefAttr>>(defOp)
                           .Case([&](P4HIR::ReadOp readOp) -> FailureOr<SymbolRefAttr> {
                               auto symRefOp =
                                   readOp.getRef().getDefiningOp<BMv2IR::SymToValueOp>();
                               if (!symRefOp) {
                                   return readOp.emitError("Expected symref op");
                               }
                               return symRefOp.getDecl();
                           })
                           .Default([](Operation *op) {
                               return op->emitError("Unsupported op for emit header");
                           });
            if (failed(ref)) return WalkResult::interrupt();
            headersRef.push_back(ref.value());
            return WalkResult::advance();
        });
        if (walkRes.wasInterrupted()) return failure();
        rewriter.replaceOpWithNewOp<BMv2IR::DeparserOp>(op, op.getSymName(),
                                                        rewriter.getArrayAttr(headersRef));
        return success();
    }
};

struct CalculationConversionPattern : public OpConversionPattern<P4HIR::ControlOp> {
    using OpConversionPattern<P4HIR::ControlOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ControlOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto isCalculation = BMv2IR::isCalculationControl(op);
        if (failed(isCalculation)) return failure();
        if (!isCalculation.value()) return failure();
        auto controlApply = cast<P4HIR::ControlApplyOp>(op.getBody().front().getTerminator());
        auto walkRes = controlApply.walk([&](P4HIR::CallOp callOp) {
            if (failed(processCall(callOp, rewriter))) return WalkResult::interrupt();
            return WalkResult::advance();
        });
        if (walkRes.wasInterrupted()) return failure();
        rewriter.eraseOp(op);
        return success();
    }

 private:
    static LogicalResult getExpressionForInput(Location loc, Value val,
                                               SmallVector<Operation *> &ops) {
        auto defOp = val.getDefiningOp();
        if (!defOp) return emitError(loc, "Expected defining op");
        ops.push_back(defOp);

        if (isa<BMv2IR::SymToValueOp>(defOp) || isa<P4HIR::ConstOp>(defOp)) return success();
        for (auto &operand : defOp->getOpOperands()) {
            if (failed(getExpressionForInput(loc, operand.get(), ops))) return failure();
        }
        return success();
    }

    static FailureOr<BMv2IR::CalculationOp> handleVerifyOrUpdateChecksumCall(
        P4HIR::CallOp callOp, PatternRewriter &rewriter) {
        auto name = BMv2IR::getUniqueNameInParentModule(callOp, "calculation");
        PatternRewriter::InsertionGuard guard(rewriter);
        auto args = callOp.getArgOperands();
        if (args.size() != 4) return callOp.emitError("Unexpected number of args");
        auto algorithmConstOp = args[3].getDefiningOp<P4HIR::ConstOp>();
        if (!algorithmConstOp) return callOp.emitError("Expected ConstOp as fourth argument");
        auto algoEnumField = algorithmConstOp.getValueAs<P4HIR::EnumFieldAttr>();
        if (!algoEnumField)
            return callOp.emitError("Expected algorithm to come from an Enum field");
        auto algoName = algoEnumField.getField();
        auto calcOp = rewriter.create<BMv2IR::CalculationOp>(callOp.getLoc(), name, algoName);
        auto &calcBlock = calcOp.getInputsRegion().emplaceBlock();
        auto inputTuple = args[1].getDefiningOp<P4HIR::TupleOp>();
        if (!inputTuple) return callOp.emitError("Expected tuple as second arg");
        SmallVector<Value> calcArgs;
        for (auto input : inputTuple.getInput()) {
            calcArgs.push_back(input);
            SmallVector<Operation *> definingOps;
            if (failed(getExpressionForInput(callOp.getLoc(), input, definingOps)))
                return failure();
            for (auto op : definingOps) {
                rewriter.moveOpBefore(op, &calcBlock, calcBlock.end());
            }
        }
        rewriter.setInsertionPointToEnd(&calcBlock);
        rewriter.create<BMv2IR::YieldOp>(callOp.getLoc(), calcArgs);

        return calcOp;
    }

    static FailureOr<std::pair<SymbolRefAttr, StringAttr>> getVerifyTargetField(
        P4HIR::CallOp callOp, PatternRewriter &rewriter) {
        // We expect the target field for the checksum to come from a StructExtractOp
        auto args = callOp.getArgOperands();
        if (args.size() != 4) return callOp.emitError("Expected 4 arguments to verify_checksum");
        auto extract = args[2].getDefiningOp<P4HIR::StructExtractOp>();
        if (!extract)
            return callOp.emitError(
                "Expected verify_checksum target field to come from an extract op");
        auto read = extract.getInput().getDefiningOp<P4HIR::ReadOp>();
        if (!read) return callOp.emitError("Expected extract input to come from a read op");
        auto symRef = read.getRef().getDefiningOp<BMv2IR::SymToValueOp>();
        if (!symRef) return callOp.emitError("Expected read input to be an Header Instance");
        return {{symRef.getDecl(), extract.getFieldNameAttr()}};
    }

    static FailureOr<std::pair<SymbolRefAttr, StringAttr>> getUpdateTargetField(
        P4HIR::CallOp callOp, PatternRewriter &rewriter) {
        // This pattern is pretty complex since the target field for the update_checksum function is
        // `inout`, e.g.

        //  %computeChecksum0_ipv4_20 = bmv2ir.symbol_ref @computeChecksum0_ipv4 :
        //  !p4hir.ref<!ipv4_t> %hdrChecksum_field_ref = p4hir.struct_field_ref
        //  %computeChecksum0_ipv4_20["hdrChecksum"] : <!ipv4_t> %checksum_inout_arg =
        //  p4hir.variable ["checksum_inout_arg", init] : <!b16i> %val_21 = p4hir.read
        //  %hdrChecksum_field_ref : <!b16i> p4hir.assign %val_21, %checksum_inout_arg : <!b16i>
        //  p4hir.call @update_checksum (%true, %tuple, %checksum_inout_arg, %HashAlgorithm_csum16)
        //  %val_22 = p4hir.read %checksum_inout_arg : <!b16i>
        //  p4hir.assign %val_22, %hdrChecksum_field_ref : <!b16i>

        // so we need to match to try and find the field_ref op for the target.
        // This pattern could be greatly simplified once we implement copy elision for inout field
        // args

        auto args = callOp.getArgOperands();
        if (args.size() != 4) return callOp.emitError("Expected 4 arguments to verify_checksum");
        auto var = args[2].getDefiningOp<P4HIR::VariableOp>();
        if (!var) return callOp.emitError("Expected var op");
        P4HIR::StructFieldRefOp fieldRef = nullptr;
        for (auto user : var->getUsers()) {
            auto assignCandidate = dyn_cast<P4HIR::AssignOp>(user);
            if (!assignCandidate) continue;
            auto readLhs = assignCandidate.getValue().getDefiningOp<P4HIR::ReadOp>();
            auto ref = assignCandidate.getRef().getDefiningOp();
            if (!readLhs || !ref || ref != var) continue;
            auto fieldRefCandidate = readLhs.getRef().getDefiningOp<P4HIR::StructFieldRefOp>();
            if (!fieldRefCandidate) continue;
            fieldRef = fieldRefCandidate;
        }
        if (!fieldRef) return callOp.emitError("Couldn't find field ref op for target field");
        auto symRef = fieldRef.getInput().getDefiningOp<BMv2IR::SymToValueOp>();
        if (!symRef) return callOp.emitError("Expected field ref input to be an Header Instance");

        return {{symRef.getDecl(), fieldRef.getFieldNameAttr()}};
    }

    static FailureOr<std::pair<SymbolRefAttr, StringAttr>> getTargetField(
        P4HIR::CallOp callOp, PatternRewriter &rewriter) {
        return llvm::StringSwitch<function_ref<FailureOr<std::pair<SymbolRefAttr, StringAttr>>(
            P4HIR::CallOp, PatternRewriter &)>>(callOp.getCallee().getLeafReference())
            .Case("verify_checksum", getVerifyTargetField)
            .Case("update_checksum", getUpdateTargetField)
            .Default([](P4HIR::CallOp callOp, PatternRewriter &) {
                return callOp.emitError("Unhandled function call");
            })(callOp, rewriter);
    }

    static LogicalResult processCall(P4HIR::CallOp callOp, PatternRewriter &rewriter) {
        auto calleeName = callOp.getCallee().getLeafReference();
        // Generate the calculation
        auto maybeCalcOp =
            llvm::StringSwitch<
                function_ref<FailureOr<BMv2IR::CalculationOp>(P4HIR::CallOp, PatternRewriter &)>>(
                calleeName)
                .Case("verify_checksum", handleVerifyOrUpdateChecksumCall)
                .Case("update_checksum", handleVerifyOrUpdateChecksumCall)
                .Default([](P4HIR::CallOp callOp, PatternRewriter &) {
                    return callOp.emitError("Unhandled function call");
                })(callOp, rewriter);
        if (failed(maybeCalcOp)) return failure();

        // Generate the checksum
        auto name = BMv2IR::getUniqueNameInParentModule(callOp, "checksum");
        auto maybeFieldRef = getTargetField(callOp, rewriter);
        if (failed(maybeFieldRef)) return failure();
        auto [headerRef, fieldName] = maybeFieldRef.value();
        bool isUpdate = llvm::StringSwitch<bool>(calleeName)
                            .Case("verify_checksum", false)
                            .Case("update_checksum", true);
        auto type = rewriter.getStringAttr("generic");
        auto calcRef = SymbolRefAttr::get(maybeCalcOp->getSymNameAttr());
        auto checksum = rewriter.create<BMv2IR::ChecksumOp>(callOp.getLoc(), name, headerRef,
                                                            fieldName, type, calcRef, isUpdate);
        auto &checksumBlock = checksum.getIfCondRegion().emplaceBlock();
        SmallVector<Operation *> definingOps;
        auto args = callOp.getArgOperands();
        if (!isa<P4HIR::BoolType>(args[0].getType()))
            return callOp.emitError("Expected boolean arg");
        if (failed(getExpressionForInput(callOp.getLoc(), args[0], definingOps))) return failure();
        for (auto op : definingOps) rewriter.moveOpBefore(op, &checksumBlock, checksumBlock.end());
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&checksumBlock);
        rewriter.create<BMv2IR::YieldOp>(callOp.getLoc(), ValueRange{args[0]});

        return success();
    }
};

struct RemovePackageInstantiationPattern : public OpConversionPattern<BMv2IR::V1SwitchOp> {
    using OpConversionPattern<BMv2IR::V1SwitchOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BMv2IR::V1SwitchOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// At this point in the pipeline all the "dataflow" goes through HeaderInstances, so we can safely
// remove ScopeOp
struct RemoveScopePattern : public OpConversionPattern<P4HIR::ScopeOp> {
    using OpConversionPattern<P4HIR::ScopeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ScopeOp op, OpAdaptor operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto &block = op.getScopeRegion().front();
        // Remove terminating yield
        rewriter.eraseOp(block.getTerminator());

        rewriter.inlineBlockBefore(&block, op);
        rewriter.eraseOp(op);
        return success();
    }
};

struct P4HIRToBMv2IRPass : public P4::P4MLIR::impl::P4HIRToBmv2IRBase<P4HIRToBMv2IRPass> {
    void runOnOperation() override {
        MLIRContext &context = getContext();
        mlir::ModuleOp module = getOperation();

        ConversionTarget target(context);
        RewritePatternSet patterns(&context);
        P4HIRToBMv2IRTypeConverter converter;
        patterns
            .add<HeaderInstanceOpConversionPattern, ParserOpConversionPattern,
                 ParserStateOpConversionPattern, ExtractOpConversionPattern,
                 ExtractVLOpConversionPattern, AssignOpToAssignHeaderPattern, AssignOpPattern,
                 ReadOpConversionPattern, FieldRefConversionPattern,
                 StructExtractOpConversionPattern, SymToValConversionPattern,
                 CompareValidityToD2BPattern, PipelineConversionPattern, DeparserConversionPattern,
                 CalculationConversionPattern, RemovePackageInstantiationPattern,
                 RemoveScopePattern, AssignOpToValidityPattern, StructAssignOpPattern>(converter,
                                                                                       &context);

        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        auto isHeaderOrRef = [](Type ty) {
            if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
                return isa<BMv2IR::HeaderType>(refTy.getObjectType());
            return isa<BMv2IR::HeaderType>(ty);
        };
        target.addDynamicallyLegalOp<BMv2IR::HeaderInstanceOp>(
            [&](BMv2IR::HeaderInstanceOp headerInstanceOp) {
                return isHeaderOrRef(headerInstanceOp.getHeaderType()) &&
                       headerInstanceOp.getSymName() != BMv2IR::standardMetadataOldStructName;
            });
        target.addDynamicallyLegalOp<BMv2IR::SymToValueOp>(
            [&](BMv2IR::SymToValueOp op) { return converter.isLegal(op); });
        target.addIllegalOp<P4HIR::ParserOp>();
        target.addIllegalOp<P4HIR::ParserStateOp>();
        target.addIllegalOp<P4CoreLib::PacketExtractOp>();
        target.addIllegalOp<P4CoreLib::PacketExtractVariableOp>();
        target.addIllegalOp<P4HIR::AssignOp>();
        target.addIllegalOp<P4HIR::StructFieldRefOp>();
        target.addIllegalOp<P4HIR::ReadOp>();

        target.addDynamicallyLegalOp<P4HIR::CmpOp>([](P4HIR::CmpOp cmpOp) {
            auto cmpKind = cmpOp.getKind();
            if (cmpKind != P4HIR::CmpOpKind::Eq) return true;
            bool isValidLhs = isValid(cmpOp.getLhs());
            bool isValidRhs = isValid(cmpOp.getRhs());
            return !isValidLhs && !isValidRhs;
        });
        target.addIllegalOp<P4HIR::ControlOp>();
        target.addIllegalOp<P4HIR::StructExtractOp>();
        target.addIllegalOp<BMv2IR::V1SwitchOp>();
        target.addIllegalOp<P4HIR::ScopeOp>();

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
