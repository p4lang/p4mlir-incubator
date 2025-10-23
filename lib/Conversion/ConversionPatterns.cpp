#include "p4mlir/Conversion/ConversionPatterns.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define DEBUG_TYPE "p4hir-conversion-patterns"

using namespace mlir;

using namespace P4::P4MLIR;

namespace {
struct AnyCallOpInterfaceConversionPattern : public OpInterfaceConversionPattern<CallOpInterface> {
    using OpInterfaceConversionPattern<CallOpInterface>::OpInterfaceConversionPattern;

    LogicalResult matchAndRewrite(CallOpInterface callOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        FailureOr<Operation *> newOp =
            doTypeConversion(callOp, operands, rewriter, getTypeConverter());
        if (failed(newOp)) return failure();

        return success();
    }
};

}  // end anonymous namespace

P4HIRTypeConverter::P4HIRTypeConverter() {
    addConversion([&](mlir::Type t) { return t; });

    addTypeAttributeConversion([](mlir::Type, Attribute attr) { return attr; });

    addTypeAttributeConversion([&](mlir::Type type, P4HIR::AggAttr attr) {
        if (isLegal(type)) return attr;

        auto newAttrs = llvm::map_to_vector(
            attr.getFields().getAsRange<mlir::TypedAttr>(), [&](auto fieldAttr) {
                return convertTypeAttribute(fieldAttr.getType(), fieldAttr).value_or(nullptr);
            });
        return P4HIR::AggAttr::get(convertType(type), ArrayAttr::get(attr.getContext(), newAttrs));
    });

    addTypeAttributeConversion([&](mlir::Type type, P4HIR::SetAttr attr) {
        if (isLegal(type)) return attr;

        auto newMembers = llvm::map_to_vector(
            attr.getMembers().getAsRange<mlir::TypedAttr>(), [&](auto memberAttr) {
                return convertTypeAttribute(memberAttr.getType(), memberAttr).value_or(nullptr);
            });
        return P4HIR::SetAttr::get(mlir::cast<P4HIR::SetType>(convertType(type)), attr.getKind(),
                                   ArrayAttr::get(attr.getContext(), newMembers));
    });

    addConversion([&](P4HIR::CtorType ctorType) {
        // Expect empty ctor args
        return P4HIR::CtorType::get(ctorType.getContext(), ctorType.getInputs(),
                                    convertType(ctorType.getReturnType()));
    });

    addConversion([&](P4HIR::ExternType externType) -> mlir::Type {
        SmallVector<mlir::Type> newTypeArgs;
        if (failed(convertTypes(externType.getTypeArguments(), newTypeArgs))) return nullptr;
        return P4HIR::ExternType::get(externType.getContext(), externType.getName(), newTypeArgs,
                                      externType.getAnnotations());
    });

    addConversion([&](P4HIR::ControlType controlType) -> mlir::Type {
        SmallVector<mlir::Type> newInputs, newTypeArgs;
        if (failed(convertTypes(controlType.getInputs(), newInputs))) return nullptr;
        if (failed(convertTypes(controlType.getTypeArguments(), newTypeArgs))) return nullptr;

        return P4HIR::ControlType::get(controlType.getContext(), controlType.getName(), newInputs,
                                       newTypeArgs, controlType.getAnnotations());
    });

    addConversion([&](P4HIR::ParserType parserType) -> mlir::Type {
        SmallVector<mlir::Type> newInputs, newTypeArgs;
        if (failed(convertTypes(parserType.getInputs(), newInputs))) return nullptr;
        if (failed(convertTypes(parserType.getTypeArguments(), newTypeArgs))) return nullptr;

        return P4HIR::ParserType::get(parserType.getContext(), parserType.getName(), newInputs,
                                      newTypeArgs, parserType.getAnnotations());
    });

    addConversion([&](P4HIR::SetType setType) {
        return P4HIR::SetType::get(convertType(setType.getElementType()));
    });

    addConversion([&](P4HIR::ArrayType arrayType) {
        return P4HIR::ArrayType::get(arrayType.getSize(), convertType(arrayType.getElementType()));
    });

    addConversion([&](P4HIR::HeaderStackType hsType) {
        return P4HIR::HeaderStackType::get(
            hsType.getContext(), hsType.getArraySize(),
            cast<P4HIR::StructLikeTypeInterface>(convertType(hsType.getArrayElementType())));
    });

    addConversion([&](P4HIR::HeaderUnionType headerunionType) {
        SmallVector<P4HIR::FieldInfo> newFields;
        llvm::transform(headerunionType.getFields(), std::back_inserter(newFields),
                        [&](auto field) -> P4HIR::FieldInfo {
                            return {field.name, convertType(field.type), field.annotations};
                        });

        return P4HIR::HeaderUnionType::get(headerunionType.getContext(), headerunionType.getName(),
                                           newFields, headerunionType.getAnnotations());
    });

    addConversion([&](P4HIR::HeaderType headerType) {
        SmallVector<P4HIR::FieldInfo> newFields;
        llvm::transform(headerType.getFields(), std::back_inserter(newFields),
                        [&](auto field) -> P4HIR::FieldInfo {
                            return {field.name, convertType(field.type), field.annotations};
                        });
        // Remove validity bit field, it is added automatically
        newFields.pop_back();

        return P4HIR::HeaderType::get(headerType.getContext(), headerType.getName(), newFields,
                                      headerType.getAnnotations());
    });

    addConversion([&](P4HIR::StructType structType) {
        SmallVector<P4HIR::FieldInfo> newFields;
        llvm::transform(structType.getFields(), std::back_inserter(newFields),
                        [&](auto field) -> P4HIR::FieldInfo {
                            return {field.name, convertType(field.type), field.annotations};
                        });

        return P4HIR::StructType::get(structType.getContext(), structType.getName(), newFields,
                                      structType.getAnnotations());
    });

    addConversion([&](P4HIR::FuncType fnType) -> mlir::Type {
        // Convert the original function types.
        llvm::SmallVector<Type> newResults, newArguments, newTypeArgs;
        if (failed(convertTypes(fnType.getReturnTypes(), newResults))) return nullptr;
        if (failed(convertTypes(fnType.getInputs(), newArguments))) return nullptr;
        if (failed(convertTypes(fnType.getTypeArguments(), newTypeArgs))) return nullptr;

        return P4HIR::FuncType::get(fnType.getContext(), newArguments,
                                    newResults.empty() ? mlir::Type() : newResults.front(),
                                    newTypeArgs);
    });

    addConversion([&](P4HIR::AliasType aliasType) {
        return P4HIR::AliasType::get(aliasType.getName(), convertType(aliasType.getAliasedType()),
                                     aliasType.getAnnotations());
    });

    addConversion([&](P4HIR::ReferenceType refType) {
        return P4HIR::ReferenceType::get(convertType(refType.getObjectType()));
    });
}

static FailureOr<mlir::Attribute> convertAttribute(mlir::Attribute attr,
                                                   const TypeConverter *typeConverter) {
    return llvm::TypeSwitch<mlir::Attribute, FailureOr<mlir::Attribute>>(attr)
        .Case<mlir::TypeAttr>([&](auto typeAttr) -> FailureOr<mlir::Attribute> {
            auto newType = typeConverter->convertType(typeAttr.getValue());
            if (!newType) return failure();
            return TypeAttr::get(newType);
        })
        .Case<TypedAttr>([&](auto typedAttr) -> FailureOr<mlir::Attribute> {
            if (typeConverter->isLegal(typedAttr.getType())) return attr;

            auto newAttr = typeConverter->convertTypeAttribute(typedAttr.getType(), typedAttr);
            if (!newAttr) return failure();
            return newAttr.value();
        })
        .Case<ArrayAttr>([&](auto arrayAttr) -> FailureOr<mlir::Attribute> {
            SmallVector<Attribute> newValues;
            for (auto entry : arrayAttr) {
                auto maybeConverted = convertAttribute(entry, typeConverter);
                if (failed(maybeConverted)) return failure();
                newValues.push_back(maybeConverted.value());
            }
            return mlir::ArrayAttr::get(arrayAttr.getContext(), newValues);
        })
        .Case<DictionaryAttr>([&](auto dictAttr) -> FailureOr<mlir::Attribute> {
            SmallVector<NamedAttribute> newValues;
            for (auto entry : dictAttr) {
                auto maybeConverted = convertAttribute(entry.getValue(), typeConverter);
                if (failed(maybeConverted)) return failure();
                newValues.emplace_back(entry.getName(), maybeConverted.value());
            }
            return mlir::DictionaryAttr::get(dictAttr.getContext(), newValues);
        })

        .Default([](auto attr) { return attr; });
}

FailureOr<Operation *> P4::P4MLIR::doTypeConversion(Operation *op, ValueRange operands,
                                                    ConversionPatternRewriter &rewriter,
                                                    const TypeConverter *typeConverter) {
    // Convert the attributes.
    llvm::SmallVector<NamedAttribute, 4> newAttrs;
    newAttrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs()) {
        auto maybeNewAttr = convertAttribute(attr.getValue(), typeConverter);
        if (failed(maybeNewAttr))
            return rewriter.notifyMatchFailure(
                op->getLoc(), "failed to convert attribute '" + attr.getName().getValue() + "'");
        newAttrs.emplace_back(attr.getName(), maybeNewAttr.value());
    }

    // Convert the result types.
    llvm::SmallVector<Type, 4> newResults;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), newResults)))
        return rewriter.notifyMatchFailure(op->getLoc(), "op result type conversion failed");

    // Build the state for the edited clone.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands, newResults, newAttrs,
                         op->getSuccessors());
    for (size_t i = 0, e = op->getNumRegions(); i < e; ++i) state.addRegion();

    // Must create the op before running any modifications on the regions so that
    // we don't crash with '-debug' and so we have something to 'root update'.
    Operation *newOp = rewriter.create(state);

    // Move the regions over, converting the signatures as we go.
    rewriter.startOpModification(newOp);
    for (size_t i = 0, e = op->getNumRegions(); i < e; ++i) {
        Region &region = op->getRegion(i);
        Region *newRegion = &newOp->getRegion(i);

        // Move the region and convert the region args.
        rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
        TypeConverter::SignatureConversion result(newRegion->getNumArguments());
        if (failed(typeConverter->convertSignatureArgs(newRegion->getArgumentTypes(), result)))
            return rewriter.notifyMatchFailure(op->getLoc(),
                                               "op region signature arg conversion failed");
        if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter, &result)))
            return rewriter.notifyMatchFailure(op->getLoc(), "op region types conversion failed");
    }
    rewriter.finalizeOpModification(newOp);

    rewriter.replaceOp(op, newOp->getResults());
    return newOp;
}

void P4::P4MLIR::populateP4HIRAnyCallOpTypeConversionPattern(mlir::RewritePatternSet &patterns,
                                                             const mlir::TypeConverter &converter) {
    patterns.add<AnyCallOpInterfaceConversionPattern>(converter, patterns.getContext());
}
