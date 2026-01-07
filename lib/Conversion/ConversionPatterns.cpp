#include "p4mlir/Conversion/ConversionPatterns.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Dialect/BMv2/BMv2Ops.h"   // ===== STEP 6.1 (added)

#define DEBUG_TYPE "p4hir-conversion-patterns"

using namespace mlir;
using namespace P4::P4MLIR;

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

    addTypeAttributeConversion([&](mlir::Type type, P4HIR::EnumFieldAttr attr) {
        if (isLegal(type)) return attr;
        return P4HIR::EnumFieldAttr::get(convertType(type), attr.getField());
    });

    addConversion([&](P4HIR::CtorType ctorType) {
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
        newFields.pop_back(); // remove validity bit
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
        .Default([](auto attr) { return attr; });
}

void P4::P4MLIR::configureUnknownOpDynamicallyLegalByTypes(
    mlir::ConversionTarget &target, const mlir::TypeConverter &converter) {
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (!converter.isLegal(op->getOperandTypes()) ||
            !converter.isLegal(op->getResultTypes()))
            return false;
        return true;
    });
}

namespace {
struct LowerV1SwitchPattern
    : public mlir::OpRewritePattern<P4HIR::InstantiateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      P4HIR::InstantiateOp op,
      mlir::PatternRewriter &rewriter) const override {

    if (op.getTargetName() != "V1Switch")
      return mlir::failure();

    auto operands = op.getOperands();

    rewriter.replaceOpWithNewOp<p4mlir::bmv2::V1SwitchOp>(
        op,
        operands[0], // parser
        operands[1], // verify
        operands[2], // ingress
        operands[3], // egress
        operands[4], // update
        operands[5]  // deparser
    );

    return mlir::success();
  }
};
} // namespace
