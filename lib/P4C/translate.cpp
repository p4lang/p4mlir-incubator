// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/P4C/translate.h"

#include <algorithm>
#include <climits>

#include "p4mlir/P4C/type_converter.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/methodInstance.h"
#include "frontends/p4/parameterSubstitution.h"
#include "frontends/p4/typeMap.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/big_int.h"
#include "lib/indent.h"
#include "lib/log.h"
#include "lib/rtti_utils.h"
#include "lib/source_file.h"
#pragma GCC diagnostic pop

#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

using namespace P4::P4MLIR;

namespace {

mlir::Location getEndLoc(mlir::OpBuilder &builder, const P4::IR::Node *node) {
    CHECK_NULL(node);
    auto sourceInfo = node->getSourceInfo();
    if (!sourceInfo.isValid()) return mlir::UnknownLoc::get(builder.getContext());

    const auto &end = sourceInfo.getEnd();

    return mlir::FileLineColLoc::get(
        builder.getStringAttr(sourceInfo.getSourceFile().string_view()), end.getLineNumber(),
        end.getColumnNumber());
}

mlir::APInt toAPInt(const P4::big_int &value, unsigned bitWidth = UINT_MAX) {
    std::vector<uint64_t> valueBits;
    // Export absolute value into 64-bit unsigned values, most significant bit last
    export_bits(value, std::back_inserter(valueBits), 64, false);

    if (bitWidth == UINT_MAX) bitWidth = valueBits.size() * sizeof(valueBits[0]) * CHAR_BIT;

    mlir::APInt apValue(bitWidth, valueBits);
    if (value < 0) apValue.negate();

    return apValue;
}

class ConversionTracer {
 public:
    ConversionTracer(const char *Kind, const P4::IR::Node *node) {
        // TODO: Add TimeTrace here
        LOG4(P4::IndentCtl::indent << Kind << dbp(node) << (LOGGING(5) ? ":" : ""));
        LOG5(node);
    }
    ~ConversionTracer() { LOG4_UNINDENT; }
};
}  // namespace

// Converts P4 SourceLocation stored in 'node' into its MLIR counterpart
mlir::Location P4HIRConverter::getLoc(const P4::IR::Node *node) {
    CHECK_NULL(node);

    auto sourceInfo = node->getSourceInfo();
    if (!sourceInfo.isValid()) return mlir::UnknownLoc::get(builder.getContext());

    P4::cstring fileName = sourceInfo.getSourceFile();
    const auto &start = sourceInfo.getStart();
    const auto &end = sourceInfo.getEnd();
    const auto &posStart = sourceInfo.toPosition();
    const auto &posEnd = sourceInfo.toPositionEnd();

    // TODO: This is actually not correct as we are mixing original file lines
    // (before preprocessor) with physical columns (after preprocessor), but p4c
    // does not give / track original columns at all...
    return mlir::FileLineColRange::get(builder.getStringAttr(fileName.string_view()),
                                       posStart.sourceLine, start.getColumnNumber() + 1,
                                       posEnd.sourceLine, end.getColumnNumber() + 1);
}

mlir::Type P4HIRConverter::getOrCreateType(const P4::IR::Type *type) {
    P4TypeConverter cvt(*this);
    type->apply(cvt, getChildContext());
    return cvt.getType();
}

mlir::Type P4HIRConverter::getOrCreateType(const P4::IR::Expression *expr) {
    return getOrCreateType(typeMap->getType(expr, true));
}

mlir::Type P4HIRConverter::getOrCreateConstructorType(const P4::IR::Type_Method *type) {
    // These things are a bit special: we keep names to simplify further
    // specialization during instantiation
    if (auto convertedType = findType(type)) return convertedType;

    ConversionTracer trace("Converting ctor type ", type);
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>, 4> argTypes;

    CHECK_NULL(type->parameters);

    mlir::Type resultType = getOrCreateType(type->returnType);

    for (const auto *param : type->parameters->parameters) {
        BUG_CHECK(param->direction == P4::IR::Direction::None, "expected directionless parameter");
        mlir::Type type = getOrCreateType(param);
        argTypes.emplace_back(builder.getStringAttr(param->name.string_view()), type);
    }

    auto mlirType = P4HIR::CtorType::get(argTypes, resultType);
    return setType(type, mlirType);
}

mlir::Value P4HIRConverter::getValueForSymbol(const P4::IR::Node *node, bool unchecked) {
    // Constants are materialized elsewhere
    if (node->is<P4::IR::Declaration_Constant>()) return {};

    if (const auto *decl = node->to<P4::IR::Declaration>()) {
        auto sym = p4Symbols.lookup(decl);
        BUG_CHECK(sym, "expected symbol '%1%' (aka %2%) to be converted", node, dbp(node));
        mlir::Type type;
        if (const auto *inst = decl->to<P4::IR::Declaration_Instance>())
            type = getOrCreateType(inst->type);
        else if (const auto *var = decl->to<P4::IR::Declaration_Variable>())
            type = getOrCreateType(var);
        else if (const auto *param = decl->to<P4::IR::Parameter>())
            type = getOrCreateType(param);
        else if (P4::RTTI::isAny<P4::IR::P4Table, P4::IR::P4Control, P4::IR::P4Parser,
                                 P4::IR::P4Action, P4::IR::Method>(decl)) {
            // This is a very special case mostly used in table properties
            type = P4HIR::UnknownType::get(context());
        }

        BUG_CHECK(type || unchecked, "unexpected symbolic reference to '%1%' (aka %2%)", node,
                  dbp(node));

        if (type) return P4HIR::SymToValueOp::create(builder, getLoc(node), type, sym);
    }

    return {};
}

mlir::Value P4HIRConverter::getValue(const P4::IR::Node *node, mlir::Type type, bool unchecked) {
    // If this is a PathExpression, resolve it
    if (const auto *pe = node->to<P4::IR::PathExpression>()) {
        const auto *target = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
        // Constants are special. We materialize them at each use. Therefore
        // their values are associates with PathExpression itself
        if (!target->is<P4::IR::Declaration_Constant>()) node = target;
    }

    mlir::Value val = p4Values->lookup(node);
    // If there is no value, then we'd need to materializer symbol's value. This is mostly
    // done for control / parser locals, so we constraint node types below
    if (!val) val = getValueForSymbol(node, unchecked);
    BUG_CHECK(val || unchecked, "expected '%1%' (aka %2%) to be converted", node, dbp(node));

    if (val && mlir::isa<P4HIR::ReferenceType>(val.getType()) &&
        (!type || !mlir::isa<P4HIR::ReferenceType>(type)))
        // Getting value out of variable involves a load.
        val = P4HIR::ReadOp::create(builder, getLoc(node), val);

    if (type && val && val.getType() != type)
        val = P4HIR::CastOp::create(builder, getLoc(node), type, val);

    return val;
}

mlir::Value P4HIRConverter::getValue(mlir::Value val, mlir::Type type) {
    if (mlir::isa<P4HIR::ReferenceType>(val.getType()))
        // Getting value out of variable involves a load.
        val = P4HIR::ReadOp::create(builder, val.getLoc(), val);

    if (type && val.getType() != type)
        val = P4HIR::CastOp::create(builder, val.getLoc(), type, val);

    return val;
}

mlir::Value P4HIRConverter::setValue(const P4::IR::Node *node, mlir::Value value) {
    if (!value) return value;

    if (LOGGING(4)) {
        std::string s;
        llvm::raw_string_ostream os(s);
        value.print(os, mlir::OpPrintingFlags().assumeVerified());
        LOG4("Converted " << dbp(node) << " -> \"" << s << "\"");
    }

    BUG_CHECK(!p4Values->count(node), "duplicate conversion of %1%", node);

    p4Values->insert(node, value);
    return value;
}

// We might reuse getOrCreateConstantExpression here, but given that annotations
// form entirely differen subset of IR, we'd resolve things slightly different
// on case-by-case basis (and we make annotations untyped by purpose). We might
// re-decide later.
mlir::Attribute P4HIRConverter::convertAnnotationExpr(const P4::IR::Expression *ann) {
    ConversionTracer trace("Converting annotation expression ", ann);

    // If this is a PathExpression, resolve it to the actual constant
    // declaration name, usualy this is a "leaf" case (e.g. match kinbd).
    if (const auto *pe = ann->to<P4::IR::PathExpression>()) {
        auto *resolved = resolvePath(pe->path, false);
        // See, if this a reference to a known symbol. FIXME: Simplify
        if (const auto *m = resolved->to<P4::IR::Method>())
            if (auto sym = p4Symbols.lookup(m)) return sym;
        if (const auto *f = resolved->to<P4::IR::Function>())
            if (auto sym = p4Symbols.lookup(f)) return sym;
        if (const auto *act = resolved->to<P4::IR::P4Action>())
            if (auto sym = p4Symbols.lookup(act)) return sym;
        if (const auto *act = resolved->to<P4::IR::P4Parser>())
            if (auto sym = p4Symbols.lookup(act)) return sym;
        if (const auto *act = resolved->to<P4::IR::P4Control>())
            if (auto sym = p4Symbols.lookup(act)) return sym;
        if (const auto *act = resolved->to<P4::IR::P4Table>())
            if (auto sym = p4Symbols.lookup(act)) return sym;

        const auto *decl = resolved->checkedTo<P4::IR::Declaration_ID>();
        if (pe->type->is<P4::IR::Type_MatchKind>())
            return P4HIR::MatchKindAttr::get(context(), decl->name.string_view());

        return builder.getStringAttr(decl->name.string_view());
    }
    if (const auto *str = ann->to<P4::IR::StringLiteral>()) {
        return builder.getStringAttr(str->value.string_view());
    }
    if (const auto *cst = ann->to<P4::IR::Constant>()) {
        mlir::APInt value = toAPInt(cst->value);
        return builder.getIntegerAttr(mlir::IntegerType::get(context(), value.getBitWidth()),
                                      value);
    }

    if (const auto *cst = ann->to<P4::IR::BoolLiteral>()) {
        mlir::APInt value = toAPInt(cst->value);
        return builder.getBoolAttr(cst->value);
    }

    if (const auto *typeNameExpr = ann->to<P4::IR::TypeNameExpression>()) {
        auto baseType = getOrCreateType(typeNameExpr->typeName);
        return mlir::TypeAttr::get(baseType);
    }

    if (const auto *lst = ann->to<P4::IR::ListExpression>()) {
        llvm::SmallVector<mlir::Attribute, 4> fields;
        for (const auto *field : lst->components) fields.push_back(convertAnnotationExpr(field));
        return builder.getArrayAttr(fields);
    }

    if (const auto *str = ann->to<P4::IR::StructExpression>()) {
        mlir::NamedAttrList fields;
        for (const auto *field : str->components)
            fields.push_back(builder.getNamedAttr(field->name.string_view(),
                                                  convertAnnotationExpr(field->expression)));
        return fields.getDictionary(context());
    }

    if (const auto *arr = ann->to<P4::IR::ArrayIndex>()) {
        auto base = mlir::cast<mlir::ArrayAttr>(convertAnnotationExpr(arr->left));
        auto idx = mlir::cast<mlir::IntegerAttr>(convertAnnotationExpr(arr->right));

        return base[idx.getInt()];
    }

    if (const auto *m = ann->to<P4::IR::Member>()) {
        if (const auto *typeNameExpr = m->expr->to<P4::IR::TypeNameExpression>()) {
            auto baseType = getOrCreateType(typeNameExpr->typeName);
            if (auto errorType = mlir::dyn_cast<P4HIR::ErrorType>(baseType))
                return P4HIR::ErrorCodeAttr::get(errorType, m->member.string_view());

            if (mlir::isa<P4HIR::EnumType, P4HIR::SerEnumType>(baseType))
                return P4HIR::EnumFieldAttr::get(baseType, m->member.string_view());

            // TODO: Do we want to introduce "StructFieldAttr" to represent
            // reference to struct field?
        }
    }

    BUG("do not know how to convert this annotation: %1%", ann);
}

mlir::Attribute P4HIRConverter::convert(const P4::IR::Annotation *ann) {
    return std::visit(
        [&](const auto &body) -> mlir::Attribute {
            using T = std::decay_t<decltype(body)>;
            if constexpr (std::is_same_v<T, P4::IR::Vector<P4::IR::Expression>>) {
                llvm::SmallVector<mlir::Attribute> fields;
                for (const auto entry : body) {
                    fields.emplace_back(convertAnnotationExpr(entry));
                }
                if (fields.empty())
                    return mlir::UnitAttr::get(context());
                else if (fields.size() == 1)
                    return fields.front();
                return mlir::ArrayAttr::get(context(), fields);
            } else if constexpr (std::is_same_v<T,
                                                P4::IR::IndexedVector<P4::IR::NamedExpression>>) {
                llvm::SmallVector<mlir::NamedAttribute> fields;
                for (const auto entry : body) {
                    fields.emplace_back(builder.getStringAttr(entry->name.string_view()),
                                        convertAnnotationExpr(entry->expression));
                }
                return mlir::DictionaryAttr::get(context(), fields);
            } else if constexpr (std::is_same_v<T, P4::IR::Vector<P4::IR::AnnotationToken>>) {
                llvm::SmallVector<mlir::Attribute> fields;
                for (const auto entry : body) {
                    fields.emplace_back(builder.getStringAttr(entry->text.string_view()));
                }
                return mlir::ArrayAttr::get(context(), fields);
            } else {
                BUG("Unexpected variant field");
            }
        },
        ann->body);
}

mlir::DictionaryAttr P4HIRConverter::convert(const P4::IR::Vector<P4::IR::Annotation> &anns) {
    // We do not want to use normal visit() functions here as we are not
    // generating code here, only attributes
    mlir::NamedAttrList annotations;
    for (const auto *ann : anns) {
        annotations.set(ann->name.string_view(), convert(ann));
    }

    return annotations.getDictionary(context());
}

// Resolve an l-value-kind expression, building access operation for each "layer".
mlir::Value P4HIRConverter::resolveReference(const P4::IR::Node *node, bool unchecked) {
    auto ref = p4Values->lookup(node);
    if (ref) return ref;

    ConversionTracer trace("Resolving reference ", node);
    auto loc = getLoc(node);
    // Check if this is a reference to a member of something we can recognize
    if (const auto *m = node->to<P4::IR::Member>()) {
        mlir::Value fieldRef;
        auto base = resolveReference(m->expr, unchecked);
        if (m->expr->type->is<P4::IR::Type_Array>()) {
            auto arrayRef = P4HIR::StructFieldRefOp::create(builder, loc, base,
                                                            P4HIR::HeaderStackType::dataFieldName);
            auto nextIndexRef = P4HIR::StructFieldRefOp::create(
                builder, loc, base, P4HIR::HeaderStackType::nextIndexFieldName);
            auto nextIndexVal = getValue(nextIndexRef);
            if (m->member == P4::IR::Type_Array::next) {
                // TODO: Insert verify() call
                fieldRef = P4HIR::ArrayElementRefOp::create(builder, loc, arrayRef, nextIndexVal);
            } else if (m->member == P4::IR::Type_Array::last) {
                auto last = P4HIR::BinOp::create(builder, loc, P4HIR::BinOpKind::Sub, nextIndexVal,
                                                 getUIntConstant(loc, 1, 32));
                // TODO: Insert verify() call
                fieldRef = P4HIR::ArrayElementRefOp::create(builder, loc, arrayRef, last);
            } else
                BUG("unknown header stack member %1% (aka %2%)", m, dbp(m));
        } else {
            if (mlir::isa<P4HIR::ReferenceType>(base.getType()))
                fieldRef =
                    P4HIR::StructFieldRefOp::create(builder, loc, base, m->member.string_view())
                        .getResult();
            else
                fieldRef =
                    P4HIR::StructExtractOp::create(builder, loc, base, m->member.string_view())
                        .getResult();
        }

        return setValue(m, fieldRef);
    } else if (const auto *a = node->to<P4::IR::ArrayIndex>()) {
        auto base = resolveReference(a->left, unchecked);
        if (a->left->type->is<P4::IR::Type_Array>()) {
            visit(a->right);
            auto arrayRef = base;
            auto arrayType = mlir::cast<P4HIR::ReferenceType>(arrayRef.getType()).getObjectType();
            if (mlir::isa<P4HIR::HeaderStackType>(arrayType))
                arrayRef = P4HIR::StructFieldRefOp::create(builder, loc, base,
                                                           P4HIR::HeaderStackType::dataFieldName)
                               .getResult();
            auto eltRef = P4HIR::ArrayElementRefOp::create(builder, loc, arrayRef,
                                                           getValue(a->right, getB32Type()));
            return setValue(a, eltRef);
        } else
            BUG("unsupported array index reference %1% (aka %2%)", node, dbp(node));
    }

    // If this is a PathExpression, resolve it to the actual declaration, usualy this
    // is a "leaf" case.
    if (const auto *pe = node->to<P4::IR::PathExpression>()) {
        const auto *target = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
        // Constants are special. We materialize them at each use. Therefore
        // their values are associates with PathExpression itself
        if (!target->is<P4::IR::Declaration_Constant>()) node = target;
    }

    ref = p4Values->lookup(node);
    if (!ref) {
        ref = getValueForSymbol(node);
        if (!ref) {
            visit(node);
            ref = p4Values->lookup(node);
        }
    }

    BUG_CHECK(ref, "expected %1% (aka %2%) to be converted", node, dbp(node));
    // The result is expected to be an l-value
    BUG_CHECK(unchecked || mlir::isa<P4HIR::ReferenceType>(ref.getType()),
              "expected reference type for node %1%", node);

    return ref;
}

mlir::TypedAttr P4HIRConverter::resolveConstant(const P4::IR::CompileTimeValue *ctv) {
    BUG("cannot resolve this constant yet %1%", ctv);
}

mlir::TypedAttr P4HIRConverter::getOrCreateConstantExpr(const P4::IR::Expression *expr) {
    if (auto cst = p4Constants.lookup(expr)) return cst;

    ConversionTracer trace("Resolving constant expression ", expr);

    // If this is a PathExpression, resolve it to the actual constant
    // declaration initializer, usualy this is a "leaf" case.
    if (const auto *pe = expr->to<P4::IR::PathExpression>()) {
        auto *cst = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration_Constant>();
        return getOrCreateConstantExpr(cst->initializer);
    }

    if (const auto *cst = expr->to<P4::IR::Constant>()) {
        auto type = getOrCreateType(cst->type);
        mlir::APInt value;
        if (auto bitType = mlir::dyn_cast<P4HIR::BitsType>(type)) {
            value = toAPInt(cst->value, bitType.getWidth());
        } else {
            value = toAPInt(cst->value);
        }

        return setConstantExpr(expr, P4HIR::IntAttr::get(context(), type, value));
    }
    if (const auto *b = expr->to<P4::IR::BoolLiteral>()) {
        // FIXME: For some reason type inference uses `Type_Unknown` for BoolLiteral (sic!)
        // auto type = mlir::cast<P4HIR::BoolType>(getOrCreateType(b->type));
        auto type = P4HIR::BoolType::get(context());

        return setConstantExpr(b, P4HIR::BoolAttr::get(context(), type, b->value));
    }
    if (const auto *s = expr->to<P4::IR::StringLiteral>()) {
        auto type = P4HIR::StringType::get(context());

        return setConstantExpr(s, mlir::StringAttr::get(s->value.string_view(), type));
    }
    if (const auto *h = expr->to<P4::IR::InvalidHeader>()) {
        auto type = mlir::cast<P4HIR::HasDefaultValue>(getOrCreateType(h->headerType));
        auto defValue = type.getDefaultValue();
        BUG_CHECK(defValue, "cannot resolve default value for %1%", expr);

        return setConstantExpr(h, defValue);
    }
    if (const auto *hu = expr->to<P4::IR::InvalidHeaderUnion>()) {
        auto type = mlir::cast<P4HIR::HasDefaultValue>(getOrCreateType(hu->headerUnionType));
        auto defValue = type.getDefaultValue();
        BUG_CHECK(defValue, "cannot resolve default value for %1%", expr);

        return setConstantExpr(hu, defValue);
    }

    if (const auto *def = expr->to<P4::IR::DefaultExpression>()) {
        return setConstantExpr(def, P4HIR::UniversalSetAttr::get(context()));
    }

    if (const auto *cast = expr->to<P4::IR::Cast>()) {
        mlir::Type destType = getOrCreateType(cast);
        mlir::Type srcType = getOrCreateType(cast->expr);
        auto srcAttr = getOrCreateConstantExpr(cast->expr);

        // Fold equal-type casts (e.g. due to typedefs)
        if (destType == srcType) return setConstantExpr(expr, srcAttr);

        // Fold some conversions
        if (auto castResult = P4HIR::foldConstantCast(destType, srcAttr))
            return setConstantExpr(expr, castResult);

        // Handle casts to aliased types
        if (auto destAliasType = mlir::dyn_cast<P4HIR::AliasType>(destType)) {
            assert(destAliasType.getAliasedType() == srcType && "expected aliased types match");
            if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(srcType)) {
                auto castee = mlir::cast<P4HIR::IntAttr>(srcAttr);
                return setConstantExpr(
                    expr, P4HIR::IntAttr::get(context(), destAliasType, castee.getValue()));
            }
            if (auto srcBoolType = mlir::dyn_cast<P4HIR::BoolType>(srcType)) {
                auto castee = mlir::cast<P4HIR::BoolAttr>(srcAttr);
                return setConstantExpr(
                    expr, P4HIR::BoolAttr::get(context(), destAliasType, castee.getValue()));
            }
        }
    }
    if (const auto *lst = expr->to<P4::IR::ListExpression>()) {
        auto type = getOrCreateType(lst->type);
        llvm::SmallVector<mlir::Attribute, 4> fields;

        // If list expression may have set components, then type unification
        // gives us a product type. Handle it separately
        if (auto setType = mlir::dyn_cast<P4HIR::SetType>(type)) {
            for (const auto *field : lst->components) {
                auto fieldConstant = getOrCreateConstantExpr(field);
                if (!mlir::isa<P4HIR::SetType>(fieldConstant.getType()))
                    fieldConstant = P4HIR::SetAttr::get(
                        P4HIR::SetType::get(fieldConstant.getType()), P4HIR::SetKind::Constant,
                        builder.getArrayAttr({fieldConstant}));
                fields.push_back(fieldConstant);
            }
            return setConstantExpr(expr, P4HIR::SetAttr::get(setType, P4HIR::SetKind::Product,
                                                             builder.getArrayAttr(fields)));
        }

        for (const auto *field : lst->components) fields.push_back(getOrCreateConstantExpr(field));
        return setConstantExpr(expr, P4HIR::AggAttr::get(type, builder.getArrayAttr(fields)));
    }
    if (const auto *str = expr->to<P4::IR::StructExpression>()) {
        auto type = getOrCreateType(str->type);
        llvm::SmallVector<mlir::Attribute, 4> fields;
        for (const auto *field : str->components)
            fields.push_back(getOrCreateConstantExpr(field->expression));
        return setConstantExpr(expr, P4HIR::AggAttr::get(type, builder.getArrayAttr(fields)));
    }
    if (const auto *arr = expr->to<P4::IR::ArrayIndex>()) {
        auto base = mlir::cast<P4HIR::AggAttr>(getOrCreateConstantExpr(arr->left));
        auto idx = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(arr->right));

        auto field = base.getFields()[idx.getUInt()];
        auto fieldType = getOrCreateType(arr->type);
        return setConstantExpr(expr, getTypedConstant(fieldType, field));
    }
    if (const auto *range = expr->to<P4::IR::Range>()) {
        auto rangeType = mlir::cast<P4HIR::SetType>(getOrCreateType(range->type));
        auto left = getOrCreateConstantExpr(range->left);
        auto right = getOrCreateConstantExpr(range->right);
        return setConstantExpr(expr, P4HIR::SetAttr::get(rangeType, P4HIR::SetKind::Range,
                                                         builder.getArrayAttr({left, right})));
    }
    if (const auto *mask = expr->to<P4::IR::Mask>()) {
        auto maskType = mlir::cast<P4HIR::SetType>(getOrCreateType(mask->type));
        auto left = getOrCreateConstantExpr(mask->left);
        auto right = getOrCreateConstantExpr(mask->right);
        return setConstantExpr(expr, P4HIR::SetAttr::get(maskType, P4HIR::SetKind::Mask,
                                                         builder.getArrayAttr({left, right})));
    }

    if (const auto *m = expr->to<P4::IR::Member>()) {
        if (const auto *typeNameExpr = m->expr->to<P4::IR::TypeNameExpression>()) {
            auto baseType = getOrCreateType(typeNameExpr->typeName);
            if (auto errorType = mlir::dyn_cast<P4HIR::ErrorType>(baseType)) {
                return setConstantExpr(
                    expr, P4HIR::ErrorCodeAttr::get(errorType, m->member.string_view()));
            }

            if (mlir::isa<P4HIR::EnumType, P4HIR::SerEnumType>(baseType))
                return setConstantExpr(
                    expr, P4HIR::EnumFieldAttr::get(baseType, m->member.string_view()));
            else
                BUG("invalid member reference %1%", m);
        }

        auto base = mlir::cast<P4HIR::AggAttr>(getOrCreateConstantExpr(m->expr));
        auto structType = mlir::cast<P4HIR::StructType>(base.getType());

        if (auto maybeIdx = structType.getFieldIndex(m->member.string_view())) {
            auto field = base.getFields()[*maybeIdx];
            auto fieldType = structType.getFieldType(m->member.string_view());

            return setConstantExpr(expr, getTypedConstant(fieldType, field));
        } else
            BUG("invalid member reference %1%", m);
    }

    if (const auto *eq = expr->to<P4::IR::Equ>()) {
        auto lhs = getOrCreateConstantExpr(eq->left);
        auto rhs = getOrCreateConstantExpr(eq->right);
        return setConstantExpr(expr, P4HIR::BoolAttr::get(context(), lhs == rhs));
    }

    if (const auto *eq = expr->to<P4::IR::Neq>()) {
        auto lhs = getOrCreateConstantExpr(eq->left);
        auto rhs = getOrCreateConstantExpr(eq->right);
        return setConstantExpr(expr, P4HIR::BoolAttr::get(context(), lhs != rhs));
    }

    if (const auto *eq = expr->to<P4::IR::Shl>()) {
        auto lhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->left));
        auto rhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->right));
        return setConstantExpr(
            expr, P4HIR::IntAttr::get(context(), lhs.getType(), lhs.getValue() << rhs.getValue()));
    }

    if (const auto *eq = expr->to<P4::IR::Shr>()) {
        auto lhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->left));
        auto rhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->right));
        auto lhsType = mlir::cast<P4HIR::BitsType>(lhs.getType());
        return setConstantExpr(
            expr, P4HIR::IntAttr::get(context(), lhs.getType(),
                                      lhsType.isSigned() ? lhs.getValue().ashr(rhs.getValue())
                                                         : lhs.getValue().lshr(rhs.getValue())));
    }

    if (const auto *eq = expr->to<P4::IR::BAnd>()) {
        auto lhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->left));
        auto rhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->right));
        return setConstantExpr(
            expr, P4HIR::IntAttr::get(context(), lhs.getType(), lhs.getValue() & rhs.getValue()));
    }

    if (const auto *eq = expr->to<P4::IR::BOr>()) {
        auto lhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->left));
        auto rhs = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(eq->right));
        return setConstantExpr(
            expr, P4HIR::IntAttr::get(context(), lhs.getType(), lhs.getValue() | rhs.getValue()));
    }

    BUG("cannot resolve this constant expression yet %1% (aka %2%)", expr, dbp(expr));
}

mlir::Value P4HIRConverter::materializeConstantExpr(const P4::IR::Expression *expr) {
    ConversionTracer trace("Materializing constant expression ", expr);

    if (auto val = getValue(expr, {}, /* unchecked */ true)) return val;

    auto init = getOrCreateConstantExpr(expr);
    auto loc = getLoc(expr);

    auto val = P4HIR::ConstOp::create(builder, loc, init);
    return setValue(expr, val);
}

mlir::Value P4HIRConverter::materializeConstantDecl(const P4::IR::Declaration_Constant *decl) {
    ConversionTracer trace("Materializing constant decl ", decl);

    auto annotations = convert(decl->annotations);

    auto init = getOrCreateConstantExpr(decl->initializer);
    auto loc = getLoc(decl);

    return P4HIR::ConstOp::create(builder, loc, init, decl->name.string_view(), annotations);
}

mlir::SymbolRefAttr P4HIRConverter::setSymbol(P4Symbol symb, mlir::SymbolRefAttr value) {
    if (!value) return value;

    if (LOGGING(4)) {
        std::string s;
        llvm::raw_string_ostream os(s);
        value.print(os);
        LOG4("Bind symbol " << s);
    }

    BUG_CHECK(!p4Symbols.count(symb), "duplicate conversion of %1%");

    p4Symbols.insert(symb, value);
    return value;
}

/// Returns fully qualified symbols, if we're nested inside parser or control
mlir::SymbolRefAttr P4HIRConverter::getQualifiedSymbolRef(mlir::Operation *op) {
    auto symName = op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());
    assert(symName && "value does not have a valid symbol name");
    return getQualifiedSymbolRef(symName);
}

mlir::SymbolRefAttr P4HIRConverter::getQualifiedSymbolRef(mlir::StringAttr attr) {
    auto leafSymbol = mlir::SymbolRefAttr::get(attr);

    const auto *ctrl = getCurrentNode<P4::IR::P4Control>();
    if (!ctrl) ctrl = findContext<P4::IR::P4Control>();
    if (ctrl) {
        auto controlSymbol = builder.getStringAttr(ctrl->name.string_view());
        return mlir::SymbolRefAttr::get(controlSymbol, {leafSymbol});
    }

    const auto *parser = getCurrentNode<P4::IR::P4Parser>();
    if (!parser) parser = findContext<P4::IR::P4Parser>();
    if (parser) {
        auto parserSymbol = builder.getStringAttr(parser->name.string_view());
        return mlir::SymbolRefAttr::get(parserSymbol, {leafSymbol});
    }

    return leafSymbol;
}

bool P4HIRConverter::preorder(const P4::IR::Type *type) {
    getOrCreateType(type);
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Program *p) {
    ValueTable values;
    p4Values = &values;

    ValueScope scope(*p4Values);
    SymbolScope symbols(p4Symbols);

    // Explicitly visit child nodes to create top-level value scope
    visit(p->objects);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::BlockStatement *block) {
    ValueScope scope(*p4Values);

    // If this is a top-level block where scope is implied (e.g. function,
    // action, certain statements) do not create explicit scope.
    if (getParent<P4::IR::BlockStatement>()) {
        auto annotations = convert(block->annotations);
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto scope = P4HIR::ScopeOp::create(
            builder, getLoc(block), annotations,
            [&](mlir::OpBuilder &, mlir::Location) {  // nothing is being yielded
                visit(block->components);
            });
        builder.setInsertionPointToEnd(&scope.getScopeRegion().back());
        P4HIR::YieldOp::create(builder, getEndLoc(builder, block));
    } else
        visit(block->components);
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Constant *decl) {
    ConversionTracer trace("Skipping constant decl ", decl);

    // We do not create global constants. Instead we do materialize them
    // at each use.

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Variable *decl) {
    ConversionTracer trace("Converting ", decl);

    auto annotations = convert(decl->annotations);

    auto type = getOrCreateType(decl);

    visit(decl->initializer);

    // TODO: Choose better insertion point for alloca (entry BB or so)
    auto var = P4HIR::VariableOp::create(builder, getLoc(decl), type, decl->name.string_view(),
                                         annotations);

    if (const auto *init = decl->initializer) {
        var.setInit(true);
        auto loc = getLoc(init);
        if (init->is<P4::IR::InvalidHeader>()) {
            // Handle special case of InvalidHeader initializer, we'd want to
            // initialize validity bit only, not the whole header
            emitHeaderValidityBitAssignOp(loc, var, P4HIR::ValidityBit::Invalid);
        } else if (init->is<P4::IR::InvalidHeaderUnion>()) {
            emitSetInvalidForAllHeaders(loc, var);
        } else {
            auto objType = llvm::cast<P4HIR::ReferenceType>(type).getObjectType();
            P4HIR::AssignOp::create(builder, loc, getValue(decl->initializer, objType), var);
        }
    }

    setValue(decl, var);

    return false;
}

void P4HIRConverter::postorder(const P4::IR::Cast *cast) {
    ConversionTracer trace("Converting ", cast);

    auto src = getValue(cast->expr);
    auto destType = getOrCreateType(cast->destType);

    setValue(cast, P4HIR::CastOp::create(builder, getLoc(cast), destType, src));
}

bool P4HIRConverter::preorder(const P4::IR::Slice *slice) {
    ConversionTracer trace("Converting ", slice);

    auto maybeRef = resolveReference(slice->e0, /* unchecked */ true);
    auto destType = getOrCreateType(slice->type);

    mlir::Value sliceVal;
    if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(maybeRef.getType());
        refType && mlir::isa<P4HIR::BitsType>(refType.getObjectType())) {
        sliceVal = P4HIR::ReadSliceOp::create(builder, getLoc(slice), destType, maybeRef,
                                              slice->getH(), slice->getL());
    } else {
        sliceVal = P4HIR::SliceOp::create(builder, getLoc(slice), destType,
                                          getValue(slice->e0, getIntType(slice->e0->type)),
                                          slice->getH(), slice->getL());
    }

    setValue(slice, sliceVal);
    return false;
}

mlir::Value P4HIRConverter::emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind) {
    auto type = getOrCreateType(unop->type);
    return P4HIR::UnaryOp::create(builder, getLoc(unop), kind, getValue(unop->expr, type));
}

mlir::Value P4HIRConverter::emitBinOp(const P4::IR::Operation_Binary *binop,
                                      P4HIR::BinOpKind kind) {
    auto type = getOrCreateType(binop->type);
    return P4HIR::BinOp::create(builder, getLoc(binop), kind, getValue(binop->left, type),
                                getValue(binop->right, type));
}

mlir::Value P4HIRConverter::emitConcatOp(const P4::IR::Concat *concatop) {
    return P4HIR::ConcatOp::create(builder, getLoc(concatop), getValue(concatop->left),
                                   getValue(concatop->right));
}

mlir::Value P4HIRConverter::emitCmp(const P4::IR::Operation_Relation *relop,
                                    P4HIR::CmpOpKind kind) {
    auto lhs = getValue(relop->left);
    auto rhs = getValue(relop->right);
    if (lhs.getType() != rhs.getType()) {
        // Handle implicit conversion from serenum to underlying type
        lhs = getValue(relop->left, getIntType(relop->left->type));
        rhs = getValue(relop->right, getIntType(relop->right->type));
    }

    return P4HIR::CmpOp::create(builder, getLoc(relop), kind, lhs, rhs);
}

#define CONVERT_UNOP(Node, Kind)                                  \
    void P4HIRConverter::postorder(const P4::IR::Node *node) {    \
        ConversionTracer trace("Converting ", node);              \
        setValue(node, emitUnOp(node, P4HIR::UnaryOpKind::Kind)); \
    }

CONVERT_UNOP(Neg, Neg)
CONVERT_UNOP(UPlus, UPlus)
CONVERT_UNOP(Cmpl, Cmpl)
CONVERT_UNOP(LNot, LNot)

#undef CONVERT_UNOP

#define CONVERT_BINOP(Node, Kind)                                \
    void P4HIRConverter::postorder(const P4::IR::Node *node) {   \
        ConversionTracer trace("Converting ", node);             \
        setValue(node, emitBinOp(node, P4HIR::BinOpKind::Kind)); \
    }

CONVERT_BINOP(Mul, Mul)
CONVERT_BINOP(Div, Div)
CONVERT_BINOP(Mod, Mod)
CONVERT_BINOP(Add, Add)
CONVERT_BINOP(Sub, Sub)
CONVERT_BINOP(AddSat, AddSat)
CONVERT_BINOP(SubSat, SubSat)
CONVERT_BINOP(BOr, Or)
CONVERT_BINOP(BAnd, And)
CONVERT_BINOP(BXor, Xor)

#undef CONVERT_BINOP

void P4HIRConverter::postorder(const P4::IR::Concat *concat) {
    ConversionTracer trace("Converting ", concat);
    setValue(concat, emitConcatOp(concat));
}

#define CONVERT_SHL_SHR_OP(P4C_Shift, PHIR_Shift)                                              \
    void P4HIRConverter::postorder(const P4::IR::P4C_Shift *op) {                              \
        ConversionTracer trace("Converting ", op);                                             \
        auto type = getOrCreateType(op->type);                                                 \
        auto intType = getIntType(op->right->type);                                            \
        auto result = P4HIR::PHIR_Shift::create(builder, getLoc(op), getValue(op->left, type), \
                                                getValue(op->right, intType));                 \
        setValue(op, result);                                                                  \
    }

CONVERT_SHL_SHR_OP(Shl, ShlOp)
CONVERT_SHL_SHR_OP(Shr, ShrOp)

#undef CONVERT_SHL_SHR_OP

#define CONVERT_CMP(Node, Kind)                                \
    void P4HIRConverter::postorder(const P4::IR::Node *node) { \
        ConversionTracer trace("Converting ", node);           \
        setValue(node, emitCmp(node, P4HIR::CmpOpKind::Kind)); \
    }

CONVERT_CMP(Equ, Eq)
CONVERT_CMP(Neq, Ne)
CONVERT_CMP(Lss, Lt)
CONVERT_CMP(Leq, Le)
CONVERT_CMP(Grt, Gt)
CONVERT_CMP(Geq, Ge)

#undef CONVERT_CMP

mlir::Value P4HIRConverter::emitValidityConstant(mlir::Location loc,
                                                 P4HIR::ValidityBit validityConstValue) {
    return P4HIR::ConstOp::create(builder, loc,
                                  P4HIR::ValidityBitAttr::get(context(), validityConstValue));
}

void P4HIRConverter::emitHeaderValidityBitAssignOp(mlir::Location loc, mlir::Value header,
                                                   P4HIR::ValidityBit validityConstValue) {
    auto validityBitConstant = emitValidityConstant(loc, validityConstValue);
    auto validityBitRef =
        P4HIR::StructFieldRefOp::create(builder, loc, header, P4HIR::HeaderType::validityBit);
    P4HIR::AssignOp::create(builder, loc, validityBitConstant, validityBitRef);
}

P4HIR::CmpOp P4HIRConverter::emitHeaderIsValidCmpOp(mlir::Location loc, mlir::Value header,
                                                    P4HIR::ValidityBit compareWith) {
    mlir::Value validityBitValue;
    if (mlir::isa<P4HIR::ReferenceType>(header.getType())) {
        auto validityBitRef =
            P4HIR::StructFieldRefOp::create(builder, loc, header, P4HIR::HeaderType::validityBit);
        validityBitValue = P4HIR::ReadOp::create(builder, loc, validityBitRef);
    } else {
        validityBitValue =
            P4HIR::StructExtractOp::create(builder, loc, header, P4HIR::HeaderType::validityBit);
    }
    auto validityConstant = emitValidityConstant(loc, compareWith);
    return P4HIR::CmpOp::create(builder, loc, P4HIR::CmpOpKind::Eq, validityBitValue,
                                validityConstant);
}

P4HIR::CmpOp P4HIRConverter::emitHeaderUnionIsValidCmpOp(mlir::Location loc,
                                                         mlir::Value headerUnion,
                                                         P4HIR::ValidityBit compareWith) {
    // Helper function to build the nested ternary operations recursively
    std::function<mlir::Value(size_t)> buildNestedTernaryOp =
        [&](size_t fieldIndex) -> mlir::Value {
        auto headerUnionType = mlir::cast<P4HIR::HeaderUnionType>(getObjectType(headerUnion));
        // If all the fields were checked, return false
        if (fieldIndex >= headerUnionType.getFields().size()) {
            return getBoolConstant(loc, false);
        }

        auto fieldInfo = headerUnionType.getFields()[fieldIndex];
        mlir::Value header;
        if (mlir::isa<P4HIR::ReferenceType>(headerUnion.getType())) {
            header = P4HIR::StructFieldRefOp::create(builder, loc, headerUnion, fieldInfo.name);
        } else {
            header = P4HIR::StructExtractOp::create(builder, loc, headerUnion, fieldInfo.name);
        }

        // Check if this member header is valid
        auto headerIsValid = emitHeaderIsValidCmpOp(loc, header, P4HIR::ValidityBit::Valid);

        // Create a ternary operation:
        // if this header is valid, return true,
        // otherwise check the next header in the header union
        auto ternaryOp = P4HIR::IfOp::create(
            builder, loc, headerIsValid.getResult(), true,
            [&](mlir::OpBuilder &b, mlir::Location loc) {
                // If this header is valid, return true
                P4HIR::YieldOp::create(b, loc, getBoolConstant(loc, true));
            },
            [&](mlir::OpBuilder &b, mlir::Location loc) {
                // If this header is not valid, check the next header
                P4HIR::YieldOp::create(b, loc, buildNestedTernaryOp(fieldIndex + 1));
            });
        return ternaryOp.getResult();
    };

    // Start the recursive building from the first field
    auto isValid = buildNestedTernaryOp(0);

    // Return a comparison operation for consistency with other validity checks
    return P4HIR::CmpOp::create(
        builder, loc, P4HIR::CmpOpKind::Eq, isValid,
        getBoolConstant(loc, compareWith == P4HIR::ValidityBit::Valid ? true : false));
}

void P4HIRConverter::emitSetInvalidForAllHeaders(mlir::Location loc, mlir::Value headerUnion,
                                                 const P4::cstring headerNameToSkip) {
    auto headerUnionType = mlir::cast<P4HIR::HeaderUnionType>(getObjectType(headerUnion));
    llvm::for_each(headerUnionType.getFields(), [&](P4HIR::FieldInfo fieldInfo) {
        if (headerNameToSkip != fieldInfo.name.getValue()) {
            auto header =
                P4HIR::StructFieldRefOp::create(builder, loc, headerUnion, fieldInfo.name);
            emitHeaderValidityBitAssignOp(loc, header, P4HIR::ValidityBit::Invalid);
        }
    });
}

mlir::Value P4HIRConverter::emitInvalidHeaderCmpOp(const P4::IR::Operation_Relation *relOp) {
    auto loc = getLoc(relOp);
    auto header = getValue(relOp->left);

    visit(relOp->left);

    if (relOp->is<P4::IR::Equ>()) {
        return emitHeaderIsValidCmpOp(loc, header, P4HIR::ValidityBit::Invalid);
    } else if (relOp->is<P4::IR::Neq>()) {
        return emitHeaderIsValidCmpOp(loc, header, P4HIR::ValidityBit::Valid);
    }
    BUG("unexpected relation operator %1%", relOp);
}

mlir::Value P4HIRConverter::emitInvalidHeaderUnionCmpOp(const P4::IR::Operation_Relation *relOp) {
    auto loc = getLoc(relOp);
    auto headerUnion = getValue(relOp->left);

    visit(relOp->left);

    if (relOp->is<P4::IR::Equ>()) {
        return emitHeaderUnionIsValidCmpOp(loc, headerUnion, P4HIR::ValidityBit::Invalid);
    } else if (relOp->is<P4::IR::Neq>()) {
        return emitHeaderUnionIsValidCmpOp(loc, headerUnion, P4HIR::ValidityBit::Valid);
    }
    BUG("unexpected relation operator %1%", relOp);
}

#define PREORDER_RELATION_OP(RelOp)                          \
    bool P4HIRConverter::preorder(const P4::IR::RelOp *op) { \
        if (op->right->is<P4::IR::InvalidHeader>()) {        \
            setValue(op, emitInvalidHeaderCmpOp(op));        \
            return false;                                    \
        }                                                    \
        if (op->right->is<P4::IR::InvalidHeaderUnion>()) {   \
            setValue(op, emitInvalidHeaderUnionCmpOp(op));   \
            return false;                                    \
        }                                                    \
        return true;                                         \
    }

PREORDER_RELATION_OP(Equ)
PREORDER_RELATION_OP(Neq)

bool P4HIRConverter::preorder(const P4::IR::AssignmentStatement *assign) {
    ConversionTracer trace("Converting ", assign);

    auto loc = getLoc(assign);

    // Assignment of InvalidHeader is special.
    if (assign->right->is<P4::IR::InvalidHeader>()) {
        const auto *member = assign->left->to<P4::IR::Member>();
        if (member != nullptr && member->expr->type->is<P4::IR::Type_HeaderUnion>()) {
            // Invalidate all headers which are the member of header union
            emitSetInvalidForAllHeaders(loc, resolveReference(member->expr));
        } else {
            // Do not materialize the whole header, assign validty bit only
            emitHeaderValidityBitAssignOp(loc, resolveReference(assign->left),
                                          P4HIR::ValidityBit::Invalid);
        }
        return false;
    }

    // Assignment of InvalidHeaderUnion is special: all headers in the header union will be set
    // to invalid
    if (assign->right->is<P4::IR::InvalidHeaderUnion>()) {
        emitSetInvalidForAllHeaders(loc, resolveReference(assign->left));
        return false;
    }

    // Invalidate all headers which are the member of header union
    if (const auto *member = assign->left->to<P4::IR::Member>()) {
        if (member->expr->type->is<P4::IR::Type_HeaderUnion>())
            emitSetInvalidForAllHeaders(loc, resolveReference(member->expr));
    }

    if (const auto *slice = assign->left->to<P4::IR::Slice>()) {
        // Fold slice of slice of slice ...
        auto expr = slice->e0;
        unsigned h = slice->getH(), l = slice->getL();
        while ((slice = expr->to<P4::IR::Slice>())) {
            int delta = slice->getL();
            expr = slice->e0;
            h += delta;
            l += delta;
        }

        auto ref = resolveReference(expr);
        P4HIR::AssignSliceOp::create(builder, loc, convert(assign->right), ref, h, l);
    } else {
        auto ref = resolveReference(assign->left);
        auto objectType = mlir::cast<P4HIR::ReferenceType>(ref.getType()).getObjectType();
        visit(assign->right);
        P4HIR::AssignOp::create(builder, loc, getValue(assign->right, objectType), ref);
    }

    return false;
}

bool P4HIRConverter::expandOpAssignBinOp(const P4::IR::OpAssignmentStatement *opAssign,
                                         P4HIR::BinOpKind kind) {
    auto loc = getLoc(opAssign);
    auto lhsRef = resolveReference(opAssign->left);
    visit(opAssign->right);

    auto type = getOrCreateType(opAssign->left->type);
    auto binop = P4HIR::BinOp::create(builder, loc, kind, getValue(opAssign->left, type),
                                      getValue(opAssign->right, type));
    P4HIR::AssignOp::create(builder, loc, binop, lhsRef);

    return false;
}

template <typename ShiftOp>
bool P4HIRConverter::expandOpAssignShift(const P4::IR::OpAssignmentStatement *opAssign) {
    auto loc = getLoc(opAssign);
    auto lhsRef = resolveReference(opAssign->left);
    visit(opAssign->right);

    auto type = getOrCreateType(opAssign->left->type);
    auto shiftop = ShiftOp::create(builder, loc, getValue(opAssign->left, type),
                                   getValue(opAssign->right, type));
    P4HIR::AssignOp::create(builder, loc, shiftop, lhsRef);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LOr *lor) {
    ConversionTracer trace("Converting ", lor);

    // Lower a || b into a ? true : b
    auto lhs = convert(lor->left);

    auto value = P4HIR::IfOp::create(
        builder, getLoc(lor), lhs, true,
        [&](mlir::OpBuilder &b, mlir::Location loc) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, lor->left), getBoolConstant(loc, true));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, lor->right), convert(lor->right));
        });

    setValue(lor, value.getResult());
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LAnd *land) {
    ConversionTracer trace("Converting ", land);

    // Lower a && b into a ? b : false
    auto lhs = convert(land->left);

    auto value = P4HIR::IfOp::create(
        builder, getLoc(land), lhs, true,
        [&](mlir::OpBuilder &b, mlir::Location) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, land->right), convert(land->right));
        },
        [&](mlir::OpBuilder &b, mlir::Location loc) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, land->left), getBoolConstant(loc, false));
        });

    setValue(land, value.getResult());
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Mux *mux) {
    ConversionTracer trace("Converting ", mux);

    // Materialize condition first
    auto cond = convert(mux->e0);

    // Make the value itself
    auto value = P4HIR::IfOp::create(
        builder, getLoc(mux), cond, true,
        [&](mlir::OpBuilder &b, mlir::Location) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, mux->e1), convert(mux->e1));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            P4HIR::YieldOp::create(b, getEndLoc(builder, mux->e2), convert(mux->e2));
        });

    setValue(mux, value.getResult());

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::IfStatement *ifs) {
    ConversionTracer trace("Converting ", ifs);

    // Materialize condition first
    auto cond = convert(ifs->condition);

    // Convert annotations, if any
    mlir::DictionaryAttr thenAnnotations, elseAnnotations;
    if (const auto *bTrue = ifs->ifTrue->to<P4::IR::BlockStatement>())
        thenAnnotations = convert(bTrue->annotations);
    if (ifs->ifFalse)
        if (const auto *bElse = ifs->ifFalse->to<P4::IR::BlockStatement>())
            elseAnnotations = convert(bElse->annotations);

    // Create if itself
    P4HIR::IfOp::create(
        builder, getLoc(ifs), cond, ifs->ifFalse,
        [&](mlir::OpBuilder &b, mlir::Location) {
            ValueScope scope(*p4Values);

            visit(ifs->ifTrue);
            P4HIR::buildTerminatedBody(b, getEndLoc(builder, ifs->ifTrue));
        },
        thenAnnotations,
        [&](mlir::OpBuilder &b, mlir::Location) {
            ValueScope scope(*p4Values);

            visit(ifs->ifFalse);
            P4HIR::buildTerminatedBody(b, getEndLoc(builder, ifs->ifFalse));
        },
        elseAnnotations);
    return false;
}

llvm::SmallVector<mlir::DictionaryAttr, 4> P4HIRConverter::convertParamAttributes(
    const P4::IR::ParameterList *params) {
    // Create attributes for directions
    llvm::SmallVector<mlir::DictionaryAttr, 4> paramsAttrs;
    for (const auto *p : params->parameters) {
        P4HIR::ParamDirection dir = P4HIR::ParamDirection::None;
        switch (p->direction) {
            case P4::IR::Direction::None:
                dir = P4HIR::ParamDirection::None;
                break;
            case P4::IR::Direction::In:
                dir = P4HIR::ParamDirection::In;
                break;
            case P4::IR::Direction::Out:
                dir = P4HIR::ParamDirection::Out;
                break;
            case P4::IR::Direction::InOut:
                dir = P4HIR::ParamDirection::InOut;
                break;
        };

        auto annotations = convert(p->annotations);
        llvm::SmallVector<mlir::NamedAttribute> paramAttrs = {
            builder.getNamedAttr(P4HIR::FuncOp::getDirectionAttrName(),
                                 P4HIR::ParamDirectionAttr::get(context(), dir)),
            builder.getNamedAttr(P4HIR::FuncOp::getParamNameAttrName(),
                                 builder.getStringAttr(p->name.string_view())),
        };
        if (!annotations.empty())
            paramAttrs.emplace_back(
                builder.getNamedAttr(P4HIR::FuncOp::getParamAnnotationAttrName(), annotations));

        paramsAttrs.emplace_back(builder.getDictionaryAttr(paramAttrs));
    }

    return paramsAttrs;
}

bool P4HIRConverter::preorder(const P4::IR::Function *f) {
    // Do not convert generic functions, these must be specialized at this point
    if (!f->type->typeParameters->empty()) return false;

    ConversionTracer trace("Converting ", f);
    ValueTable functionValues, *savedValues = p4Values;
    p4Values = &functionValues;
    ValueScope scope(functionValues);

    auto annotations = convert(f->annotations);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(f->type));
    const auto &params = f->getParameters()->parameters;

    auto argAttrs = convertParamAttributes(f->getParameters());
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto loc = getLoc(f);

    auto *parentOp = builder.getBlock()->getParentOp();
    auto origSymName = builder.getStringAttr(f->name.string_view());
    auto symName = origSymName;
    if (auto *otherOp = mlir::SymbolTable::lookupNearestSymbolFrom(parentOp, origSymName)) {
        LOG4("Function is overloaded");

        P4HIR::OverloadSetOp ovl;
        auto getUniqueName = [&](mlir::StringAttr toRename) {
            unsigned counter = 0;
            return mlir::SymbolTable::generateSymbolName<256>(
                toRename,
                [&](llvm::StringRef candidate) {
                    return ovl.lookupSymbol(builder.getStringAttr(candidate)) != nullptr;
                },
                counter);
        };

        if (auto otherFunc = llvm::dyn_cast<P4HIR::FuncOp>(otherOp)) {
            LOG4("Creating overload set");

            ovl = P4HIR::OverloadSetOp::create(builder, loc, origSymName);
            builder.setInsertionPointToStart(&ovl.createEntryBlock());
            otherFunc->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());

            // Unique the symbol name to avoid clashes in the symbol table.  The
            // overload set takes over the symbol name. Still, all the symbols
            // in `p4Symbol` are created wrt the original name, so we do not use
            // SymbolTable::rename() here.
            otherFunc.setSymName(getUniqueName(origSymName));
        } else {
            LOG4("Adding to overload set");

            ovl = llvm::cast<P4HIR::OverloadSetOp>(otherOp);
            builder.setInsertionPointToEnd(&ovl.getBody().front());
        }

        symName = builder.getStringAttr(getUniqueName(symName));
    }

    auto func = P4HIR::FuncOp::create(builder, loc, symName, funcType,
                                      /* isExternal */ false, argAttrs, annotations);
    func.createEntryBlock();

    // Iterate over parameters again binding parameter values to arguments of first BB
    auto &body = func.getBody();

    assert(body.getNumArguments() == params.size() && "invalid parameter conversion");
    for (auto [param, bodyArg] : llvm::zip(params, body.getArguments())) setValue(param, bodyArg);

    // We cannot simply visit each node of the top-level block as
    // ResolutionContext would not be able to resolve declarations there
    // (sic!)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body.front());
        visit(f->body);

        // Check if body's last block is not terminated.
        mlir::Block &b = body.back();
        if (!b.mightHaveTerminator()) {
            builder.setInsertionPointToEnd(&b);
            P4HIR::ReturnOp::create(builder, getEndLoc(builder, f));
        }
    }

    setSymbol(f, mlir::SymbolRefAttr::get(origSymName));
    p4Values = savedValues;

    return false;
}

// We treat method as an external function (w/o body)
bool P4HIRConverter::preorder(const P4::IR::Method *m) {
    ConversionTracer trace("Converting ", m);
    ValueScope scope(*p4Values);

    auto annotations = convert(m->annotations);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(m->type));

    auto argAttrs = convertParamAttributes(m->getParameters());
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto loc = getLoc(m);

    // Check if there is a declaration with the same name in the current symbol table.
    // If yes, create / add to an overload set
    auto *parentOp = builder.getBlock()->getParentOp();
    auto origSymName = builder.getStringAttr(m->name.string_view());
    auto symName = origSymName;
    if (auto *otherOp = mlir::SymbolTable::lookupNearestSymbolFrom(parentOp, origSymName)) {
        LOG4("Method is overloaded");

        P4HIR::OverloadSetOp ovl;
        auto getUniqueName = [&](mlir::StringAttr toRename) {
            unsigned counter = 0;
            return mlir::SymbolTable::generateSymbolName<256>(
                toRename,
                [&](llvm::StringRef candidate) {
                    return ovl.lookupSymbol(builder.getStringAttr(candidate)) != nullptr;
                },
                counter);
        };

        if (auto otherFunc = llvm::dyn_cast<P4HIR::FuncOp>(otherOp)) {
            LOG4("Creating overload set");

            ovl = P4HIR::OverloadSetOp::create(builder, loc, origSymName);
            builder.setInsertionPointToStart(&ovl.createEntryBlock());
            otherFunc->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());

            // Unique the symbol name to avoid clashes in the symbol table.  The
            // overload set takes over the symbol name. Still, all the symbols
            // in `p4Symbol` are created wrt the original name, so we do not use
            // SymbolTable::rename() here.
            otherFunc.setSymName(getUniqueName(origSymName));
        } else {
            LOG4("Adding to overload set");

            ovl = llvm::cast<P4HIR::OverloadSetOp>(otherOp);
            builder.setInsertionPointToEnd(&ovl.getBody().front());
        }

        symName = builder.getStringAttr(getUniqueName(symName));
    }

    P4HIR::FuncOp::create(builder, loc, symName, funcType,
                          /* isExternal */ true, argAttrs, annotations);

    setSymbol(m, mlir::SymbolRefAttr::get(origSymName));
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Action *act) {
    ConversionTracer trace("Converting ", act);

    const auto *control = findContext<P4::IR::P4Control>();
    llvm::SmallVector<mlir::TypedAttr, 2> ctorParamAttrs;
    if (control) {
        for (const auto *param : control->getConstructorParameters()->parameters) {
            auto attr = getValue(param).getDefiningOp<P4HIR::ConstOp>().getValue();
            ctorParamAttrs.push_back(attr);
        }
    }

    ValueTable actionValues, *savedValues = p4Values;
    p4Values = &actionValues;
    ValueScope scope(actionValues);

    // FIXME: Get rid of typeMap: ensure action knows its type
    auto actType = mlir::cast<P4HIR::FuncType>(getOrCreateType(typeMap->getType(act, true)));
    const auto &params = act->getParameters()->parameters;

    auto annotations = convert(act->annotations);

    auto argAttrs = convertParamAttributes(act->getParameters());
    assert(actType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto action = P4HIR::FuncOp::buildAction(builder, getLoc(act), act->name.string_view(), actType,
                                             argAttrs, annotations);

    // Iterate over parameters again binding parameter values to arguments of first BB
    auto &body = action.getBody();

    assert(body.getNumArguments() == params.size() && "invalid parameter conversion");
    for (auto [param, bodyArg] : llvm::zip(params, body.getArguments())) setValue(param, bodyArg);

    // We cannot simply visit each node of the top-level block as
    // ResolutionContext would not be able to resolve declarations there
    // (sic!)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body.front());

        if (control) {
            // Actions could refer to control's constructor arguments. Materialize them as
            // constants.
            for (auto [param, attr] :
                 llvm::zip_equal(control->getConstructorParameters()->parameters, ctorParamAttrs)) {
                llvm::StringRef paramName = param->name.string_view();
                auto val = P4HIR::ConstOp::create(builder, getLoc(param), attr, paramName);
                setValue(param, val);
            }
        }

        visit(act->body);

        // Check if body's last block is not terminated.
        mlir::Block &b = body.back();
        if (!b.mightHaveTerminator()) {
            builder.setInsertionPointToEnd(&b);
            P4HIR::ReturnOp::create(builder, getEndLoc(builder, act));
        }
    }

    // Make actions nested inside controls fully qualified, so we can resolve
    // properly even in the presence of name shadow
    setSymbol(act, getQualifiedSymbolRef(action));
    p4Values = savedValues;

    return false;
}

void P4HIRConverter::postorder(const P4::IR::ReturnStatement *ret) {
    ConversionTracer trace("Converting ", ret);

    if (ret->expression) {
        auto retVal = getValue(ret->expression);
        P4HIR::SoftReturnOp::create(builder, getLoc(ret), retVal);
    } else {
        P4HIR::SoftReturnOp::create(builder, getLoc(ret));
    }
}
void P4HIRConverter::postorder(const P4::IR::ContinueStatement *cont) {
    ConversionTracer trace("Converting ", cont);
    P4HIR::SoftContinueOp::create(builder, getLoc(cont));
}
void P4HIRConverter::postorder(const P4::IR::BreakStatement *br) {
    ConversionTracer trace("Converting ", br);
    P4HIR::SoftBreakOp::create(builder, getLoc(br));
}
void P4HIRConverter::postorder(const P4::IR::ExitStatement *ex) {
    ConversionTracer trace("Converting ", ex);

    P4HIR::ExitOp::create(builder, getLoc(ex));
}

mlir::Value P4HIRConverter::emitHeaderBuiltInMethod(mlir::Location loc,
                                                    const P4::BuiltInMethod *builtInMethod) {
    mlir::Value callResult;
    if (builtInMethod->name == P4::IR::Type_Header::setValid ||
        builtInMethod->name == P4::IR::Type_Header::setInvalid) {
        // Check if the header is a member of a header union
        if (const auto *member = builtInMethod->appliedTo->to<P4::IR::Member>()) {
            if (member->expr->type->is<P4::IR::Type_HeaderUnion>()) {
                const auto headerNameToSkip = builtInMethod->name == P4::IR::Type_Header::setValid
                                                  ? member->member.name
                                                  : nullptr;
                emitSetInvalidForAllHeaders(loc, resolveReference(member->expr), headerNameToSkip);
            }
        }

        if (builtInMethod->name == P4::IR::Type_Header::setValid) {
            emitHeaderValidityBitAssignOp(loc, resolveReference(builtInMethod->appliedTo),
                                          P4HIR::ValidityBit::Valid);
        }
    } else if (builtInMethod->name == P4::IR::Type_Header::isValid) {
        auto header = resolveReference(builtInMethod->appliedTo, /* unchecked */ true);
        return emitHeaderIsValidCmpOp(loc, header, P4HIR::ValidityBit::Valid);
    } else {
        BUG("Unsupported builtin method: %1%", builtInMethod->name);
    }

    return callResult;
}

mlir::Value P4HIRConverter::emitHeaderUnionBuiltInMethod(mlir::Location loc,
                                                         const P4::BuiltInMethod *builtInMethod) {
    if (builtInMethod->name == P4::IR::Type_Header::isValid) {
        auto headerUnion = resolveReference(builtInMethod->appliedTo, /* unchecked */ true);
        return emitHeaderUnionIsValidCmpOp(loc, headerUnion, P4HIR::ValidityBit::Valid);
    }
    BUG("Unsupported Header Union builtin method: %1%", builtInMethod->name);
}

mlir::Value P4HIRConverter::emitHeaderStackBuiltInMethod(mlir::Location loc,
                                                         const P4::BuiltInMethod *builtInMethod) {
    // Implement push_front
    // Implement pop_front
    BUG("Unsupported header stack builtin method: %1%", builtInMethod->name);
}

bool P4HIRConverter::preorder(const P4::IR::MethodCallExpression *mce) {
    ConversionTracer trace("Converting ", mce);
    const auto *instance =
        P4::MethodInstance::resolve(mce, this, typeMap, false, getChildContext());
    const auto &params = instance->originalMethodType->parameters->parameters;

    // Prepare call arguments. Note that this involves creating temporaries to
    // model copy-in/out semantics. To limit the lifetime of those temporaries, do
    // everything in the dedicated block scope. If there are no out parameters,
    // then emit everything direct.
    bool emitScope =
        std::any_of(params.begin(), params.end(), [](const auto *p) { return p->hasOut(); });
    auto convertCall = [&](mlir::OpBuilder &b, mlir::Type &resultType, mlir::Location loc) {
        // Special case: lower builtin methods.
        if (const auto *bCall = instance->to<P4::BuiltInMethod>()) {
            assert(!emitScope && "should not be inside scope");

            // TODO: Are there cases when we do not have l-value here?
            auto loc = getLoc(mce);

            // Check if this is a method call on a header or header union
            if (const auto *member = mce->method->to<P4::IR::Member>()) {
                // Check if it's a reference to a header or header union
                if (member->expr->type->is<P4::IR::Type_HeaderUnion>()) {
                    setValue(mce, emitHeaderUnionBuiltInMethod(loc, bCall));
                } else if (member->expr->type->to<P4::IR::Type_Header>()) {
                    setValue(mce, emitHeaderBuiltInMethod(loc, bCall));
                } else if (member->expr->type->to<P4::IR::Type_Array>()) {
                    setValue(mce, emitHeaderStackBuiltInMethod(loc, bCall));
                } else {
                    BUG("cannot handle this builtin method yet: %1% (aka %2%)", member,
                        dbp(member));
                }
            }
            return;
        }

        llvm::SmallVector<mlir::Value, 4> operands;
        llvm::DenseMap<const P4::IR::Argument *, mlir::Value> argValues;
        mlir::Value callResult;

        // Evaluate arguments in the call order
        for (const auto *arg : *mce->arguments) {
            ConversionTracer trace("Converting ", arg);
            mlir::Value argVal;
            // TODO: This is pretty inefficient, expose argument => parameter
            // map from ParameterSubstitution
            const auto *param = instance->substitution.findParameter(arg);

            // This is hack to support dontcare arguments (_). These have
            // Type_DontCare type, so we just create new uninitialized variable
            // of parameter type
            if (arg->expression->is<P4::IR::DefaultExpression>()) {
                auto type = getOrCreateType(param);
                auto var =
                    P4HIR::VariableOp::create(builder, getLoc(arg->expression), type, "dummy");
                setValue(arg->expression, var);
            }

            switch (auto dir = param->direction) {
                case P4::IR::Direction::None: {
                    // Add support for reference action params
                    auto paramType = getOrCreateType(param);
                    auto argType = getOrCreateType(arg->expression->type);

                    if (mlir::isa<P4HIR::ReferenceType>(paramType) &&
                        !mlir::isa<P4HIR::ReferenceType>(argType)) {
                        auto copyIn = P4HIR::VariableOp::create(
                            b, loc, P4HIR::ReferenceType::get(argType),
                            llvm::Twine(param->name.string_view()) + "_ref_arg");
                        visit(arg->expression);
                        argVal = getValue(arg->expression);

                        copyIn.setInit(true);
                        P4HIR::AssignOp::create(b, loc, argVal, copyIn);
                        argVal = copyIn;
                    } else {
                        // Nothing to do special, just pass things direct
                        visit(arg->expression);
                        argVal = getValue(arg->expression, paramType);
                    }
                    break;
                }

                case P4::IR::Direction::In: {
                    auto paramType = getOrCreateType(param);

                    // Nothing to do special, just pass things direct
                    visit(arg->expression);
                    argVal = getValue(arg->expression, paramType);
                    break;
                }
                case P4::IR::Direction::Out:
                case P4::IR::Direction::InOut: {
                    // Create temporary to hold the output value, initialize in case of inout
                    if (const auto *slice = arg->expression->to<P4::IR::Slice>()) {
                        auto sliceType = getOrCreateType(slice->type);
                        auto ref = resolveReference(slice->e0);
                        auto copyIn = P4HIR::VariableOp::create(
                            b, loc, P4HIR::ReferenceType::get(sliceType),
                            llvm::Twine(param->name.string_view()) +
                                (dir == P4::IR::Direction::InOut ? "_inout_arg" : "_out_arg"));

                        if (dir == P4::IR::Direction::InOut) {
                            copyIn.setInit(true);
                            P4HIR::AssignOp::create(
                                b, loc,
                                P4HIR::ReadSliceOp::create(b, loc, sliceType, ref, slice->getH(),
                                                           slice->getL()),
                                copyIn);
                        }
                        argVal = copyIn;
                    } else {
                        auto ref = resolveReference(arg->expression);
                        auto copyIn = P4HIR::VariableOp::create(
                            b, loc, ref.getType(),
                            llvm::Twine(param->name.string_view()) +
                                (dir == P4::IR::Direction::InOut ? "_inout_arg" : "_out_arg"));

                        if (dir == P4::IR::Direction::InOut) {
                            copyIn.setInit(true);
                            P4HIR::AssignOp::create(b, loc, P4HIR::ReadOp::create(b, loc, ref),
                                                    copyIn);
                        }
                        argVal = copyIn;
                    }

                    break;
                }
            }
            auto [it, inserted] = argValues.try_emplace(arg, argVal);
            BUG_CHECK(inserted, "duplicate conversion? %1%", it->first);
        }

        // Collect arguments in operand order
        for (const auto &param : params) {
            if (auto argument = instance->substitution.lookup(param)) {
                auto argVal = argValues.lookup(argument);
                BUG_CHECK(argVal, "unconverted argument?");

                operands.push_back(argVal);
            } else {
                // Parameter is not bound. This is possible only for actions
                // where argument might come from control plane or @optional argument.
                // Grab a suitable placeholder for it.
                mlir::Value placeholder;
                if (param->isOptional()) {
                    placeholder = P4HIR::UninitializedOp::create(builder, getLoc(mce),
                                                                 getOrCreateType(param));
                } else {
                    BUG_CHECK(param->direction == P4::IR::Direction::None,
                              "control plane values should be directionless");

                    placeholder = controlPlaneValues.lookup(param);
                    // As an extension, default-initialize all unpopulated control plane values
                    if (!placeholder) {
                        if (defaultInitialize) {
                            auto type = mlir::cast<P4HIR::HasDefaultValue>(getOrCreateType(param));
                            auto defValue = type.getDefaultValue();
                            BUG_CHECK(defValue, "cannot resolve default value for %1%", param);
                            placeholder = P4HIR::ConstOp::create(builder, getLoc(param), defValue);
                        } else {
                            BUG_CHECK(placeholder,
                                      "control plane value %1% in %2% must be populated", param,
                                      mce);
                        }
                    }
                }

                operands.push_back(placeholder);
            }
        }

        if (const auto *actCall = instance->to<P4::ActionCall>()) {
            LOG4("resolved as action call");
            auto actSym = p4Symbols.lookup(actCall->action);
            BUG_CHECK(actSym, "expected reference action to be converted: %1%", actCall->action);

            BUG_CHECK(mce->typeArguments->empty(), "expected action to be specialized");

            P4HIR::CallOp::create(b, loc, actSym, operands);
        } else if (const auto *fCall = instance->to<P4::FunctionCall>()) {
            LOG4("resolved as function call");
            auto fSym = p4Symbols.lookup(fCall->function);
            auto callResultType = getOrCreateType(instance->actualMethodType->returnType);

            BUG_CHECK(fSym, "expected reference function to be converted: %1%", fCall->function);
            BUG_CHECK(mce->typeArguments->empty(), "expected function to be specialized");

            callResult = P4HIR::CallOp::create(b, loc, fSym, callResultType, operands).getResult();
        } else if (const auto *fCall = instance->to<P4::ExternFunction>()) {
            LOG4("resolved as extern function call");
            auto fSym = p4Symbols.lookup(fCall->method);
            auto callResultType = getOrCreateType(instance->actualMethodType->returnType);

            BUG_CHECK(fSym, "expected reference function to be converted: %1%", fCall->method);

            // TODO: Move to common method
            llvm::SmallVector<mlir::Type> typeArguments;
            for (const auto *type : *mce->typeArguments) {
                typeArguments.push_back(getOrCreateType(type));
            }

            callResult =
                P4HIR::CallOp::create(b, loc, fSym, callResultType, typeArguments, operands)
                    .getResult();
        } else if (const auto *aCall = instance->to<P4::ApplyMethod>()) {
            LOG4("resolved as apply");
            BUG_CHECK(mce->typeArguments->empty(), "expected decl to be specialized");
            // Apply of something instantiated
            if (auto *decl = aCall->object->to<P4::IR::Declaration_Instance>()) {
                auto dSym = p4Symbols.lookup(decl);
                BUG_CHECK(dSym, "expected applied declaration to be converted: %1%", decl);
                P4HIR::ApplyOp::create(b, loc, dSym, operands);
            } else if (auto *table = aCall->object->to<P4::IR::P4Table>()) {
                mlir::ValueRange tableKeyArgs;
                if (auto it = tableKeyArgsMap.find(table); it != tableKeyArgsMap.end())
                    tableKeyArgs = it->second;
                auto tSym = p4Symbols.lookup(table);
                BUG_CHECK(tSym, "expected applied table to be converted: %1%", table);
                auto applyResultType = getOrCreateType(instance->actualMethodType->returnType);
                callResult =
                    P4HIR::TableApplyOp::create(b, loc, applyResultType, tSym, tableKeyArgs)
                        .getResult();
            } else
                BUG("Unsuported apply: %1% (aka %2%)", aCall->object, dbp(aCall->object));
        } else if (const auto *fCall = instance->to<P4::ExternMethod>()) {
            LOG4("resolved as extern method call ");

            // We need to do some weird dance to resolve method call, as fCall->method will not
            // resolve to a known symbol.
            const auto *member = mce->method->checkedTo<P4::IR::Member>();

            auto callResultType = getOrCreateType(instance->actualMethodType->returnType);
            auto methodName =
                mlir::SymbolRefAttr::get(builder.getContext(), member->member.string_view());
            auto externName = builder.getStringAttr(fCall->actualExternType->name.string_view());
            auto fullMethodName =
                mlir::SymbolRefAttr::get(builder.getContext(), externName, {methodName});

            // TODO: Move to common method
            llvm::SmallVector<mlir::Type> typeArguments;
            for (const auto *type : *mce->typeArguments)
                typeArguments.push_back(getOrCreateType(type));

            const P4::IR::Declaration_Instance *decl = nullptr;
            if (const auto *pe = member->expr->to<P4::IR::PathExpression>())
                decl = resolvePath(pe->path, false)->to<P4::IR::Declaration_Instance>();

            if (decl) {
                auto dSym = p4Symbols.lookup(decl);
                BUG_CHECK(dSym, "expected applied declaration to be converted: %1%", decl);

                callResult = P4HIR::CallMethodOp::create(b, loc, callResultType, dSym,
                                                         fullMethodName, typeArguments, operands)
                                 .getResult();
            } else {
                auto callee = convert(member->expr);
                callResult = P4HIR::CallMethodOp::create(b, loc, callResultType, callee,
                                                         fullMethodName, typeArguments, operands)
                                 .getResult();
            }
        } else {
            BUG("unsupported call type: %1%", mce);
        }

        for (const auto *arg : *mce->arguments) {
            // Determine the direction of the parameter
            // TODO: This is pretty inefficient, expose argument => parameter
            // map from ParameterSubstitution
            const auto *param = instance->substitution.findParameter(arg);
            if (!param->hasOut()) continue;

            auto copyOut = argValues.lookup(arg);
            BUG_CHECK(copyOut, "unconverted argument?");
            if (const auto *slice = arg->expression->to<P4::IR::Slice>()) {
                mlir::Value dest = resolveReference(slice->e0);
                P4HIR::AssignSliceOp::create(
                    b, getEndLoc(builder, mce),
                    P4HIR::ReadOp::create(builder, getEndLoc(builder, mce), copyOut), dest,
                    slice->getH(), slice->getL());
            } else {
                mlir::Value dest = resolveReference(arg->expression);
                P4HIR::AssignOp::create(
                    b, getEndLoc(builder, mce),
                    P4HIR::ReadOp::create(builder, getEndLoc(builder, mce), copyOut), dest);
            }
        }

        // If we are inside the scope, then build the yield of the call result
        if (emitScope) {
            if (callResult) {
                resultType = callResult.getType();
                P4HIR::YieldOp::create(b, getEndLoc(b, mce), callResult);
            } else
                P4HIR::YieldOp::create(b, getEndLoc(b, mce));
        } else {
            setValue(mce, callResult);
        }
    };

    if (emitScope) {
        auto scope = P4HIR::ScopeOp::create(builder, getLoc(mce), convertCall);
        setValue(mce, scope.getResults());
    } else {
        mlir::Type resultType;
        convertCall(builder, resultType, getLoc(mce));
    }

    return false;
}

// We do not need `MoveConstructors` as we resolve values directly
bool P4HIRConverter::preorder(const P4::IR::ConstructorCallExpression *cce) {
    ConversionTracer trace("Converting ", cce);

    // P4::Instantiation goes via typeMap and it returns some weird clone
    // instead of converted type
    const auto *type = resolveType(cce->constructedType);
    CHECK_NULL(type);
    LOG4("Resolved to: " << dbp(type));

    llvm::SmallVector<mlir::Value, 4> operands;
    llvm::DenseMap<const P4::IR::Argument *, mlir::Value> argValues;
    for (const auto *arg : *cce->arguments) {
        ConversionTracer trace("Converting ", arg);
        argValues.try_emplace(arg, convert(arg->expression));
    }

    auto resultType = getOrCreateType(type);

    // Resolve to base type
    if (const auto *tdef = type->to<P4::IR::Type_Typedef>()) {
        type = resolveType(tdef->type);
        CHECK_NULL(type);
        LOG4("Resolved to typedef type: " << dbp(type));
    }
    llvm::SmallVector<mlir::Type> typeParameters;
    if (const auto *spec = type->to<P4::IR::Type_Specialized>()) {
        for (const auto *type : *spec->arguments) {
            typeParameters.push_back(getOrCreateType(type));
        }
        type = resolveType(spec->baseType);
        CHECK_NULL(type);
        LOG4("Resolved to base type: " << dbp(type));
    }

    // Shuffle arguments into proper order
    auto populateOperands = [&](const P4::IR::ParameterList *params) {
        P4::ParameterSubstitution subst;
        subst.populate(params, cce->arguments);
        for (const auto &param : params->parameters) {
            auto argument = subst.lookup(param);
            auto argVal = argValues.lookup(argument);
            // Create a placeholder for @optional arguments
            if (!argVal && param->isOptional())
                argVal =
                    P4HIR::UninitializedOp::create(builder, getLoc(cce), getOrCreateType(param));
            BUG_CHECK(argVal, "unconverted argument for parameter %1%", param);
            operands.push_back(argVal);
        }
    };

    if (const auto *cont = type->to<P4::IR::IContainer>()) {
        populateOperands(cont->getConstructorParameters());
    } else {
        const auto *ext = type->checkedTo<P4::IR::Type_Extern>();
        const auto *ctor = ext->lookupConstructor(cce->arguments);
        populateOperands(ctor->getParameters());
    }

    if (const auto *parser = type->to<P4::IR::P4Parser>()) {
        LOG4("resolved as parser instantiation");
        auto parserSym = p4Symbols.lookup(parser);
        BUG_CHECK(parserSym, "expected reference parser to be converted: %1%", dbp(parser));

        auto instance = P4HIR::ConstructOp::create(builder, getLoc(cce), resultType,
                                                   parserSym.getRootReference(), operands);
        setValue(cce, instance.getResult());
    } else if (const auto *control = type->to<P4::IR::P4Control>()) {
        LOG4("resolved as control instantiation");
        auto controlSym = p4Symbols.lookup(control);
        BUG_CHECK(controlSym, "expected reference control to be converted: %1%", dbp(control));

        auto instance = P4HIR::ConstructOp::create(builder, getLoc(cce), resultType,
                                                   controlSym.getRootReference(), operands);
        setValue(cce, instance.getResult());
    } else if (const auto *ext = type->to<P4::IR::Type_Extern>()) {
        LOG4("resolved as extern instantiation");

        auto externName = builder.getStringAttr(ext->name.string_view());
        auto instance = P4HIR::ConstructOp::create(builder, getLoc(cce), resultType, externName,
                                                   operands, typeParameters);
        setValue(cce, instance.getResult());
    } else if (const auto *pkg = type->to<P4::IR::Type_Package>()) {
        LOG4("resolved as package instantiation");

        auto pkgName = builder.getStringAttr(pkg->name.string_view());
        auto instance = P4HIR::ConstructOp::create(builder, getLoc(cce), resultType, pkgName,
                                                   operands, typeParameters);
        setValue(cce, instance.getResult());
    } else {
        BUG("unsupported constructor call: %1% (of type %2%)", cce, dbp(type));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Member *m) {
    ConversionTracer trace("Converting in preorder ", m);

    // This is just enum constant
    if (const auto *typeNameExpr = m->expr->to<P4::IR::TypeNameExpression>()) {
        auto type = getOrCreateType(typeNameExpr->typeName);
        auto loc = getLoc(m);

        if (mlir::isa<P4HIR::ErrorType>(type))
            setValue(m, P4HIR::ConstOp::create(
                            builder, loc,
                            P4HIR::ErrorCodeAttr::get(type, m->member.name.string_view())));
        else if (mlir::isa<P4HIR::EnumType>(type))
            setValue(m, P4HIR::ConstOp::create(
                            builder, loc,
                            P4HIR::EnumFieldAttr::get(type, m->member.name.string_view())));
        else if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(type)) {
            setValue(m, P4HIR::ConstOp::create(
                            builder, loc,
                            P4HIR::EnumFieldAttr::get(type, m->member.name.string_view())));
        } else
            BUG("unexpected type for expression %1%", typeNameExpr);

        return false;
    }

    // Handle other members in postorder traversal
    return true;
}

void P4HIRConverter::postorder(const P4::IR::Member *m) {
    ConversionTracer trace("Converting in postorder ", m);

    // Resolve member rvalue expression to something we can reason about
    // TODO: Likely we can do similar things for the majority of struct-like
    // types
    auto parentType = getOrCreateType(m->expr);
    auto loc = getLoc(m);
    if (mlir::isa<P4HIR::StructType, P4HIR::HeaderType, P4HIR::HeaderUnionType>(parentType)) {
        // We can access to parent using struct operations
        auto parent = getValue(m->expr);
        auto field = P4HIR::StructExtractOp::create(builder, loc, parent, m->member.string_view());
        setValue(m, field);
    } else if (auto hsType = mlir::dyn_cast<P4HIR::HeaderStackType>(parentType)) {
        if (m->member == P4::IR::Type_Array::arraySize) {
            size_t sz = hsType.getArraySize();
            setValue(m, getUIntConstant(getLoc(m), sz, 32));
        } else if (m->member == P4::IR::Type_Array::lastIndex) {
            auto parent = getValue(m->expr);
            auto last = P4HIR::BinOp::create(
                builder, loc, P4HIR::BinOpKind::Sub,
                P4HIR::StructExtractOp::create(builder, loc, parent,
                                               P4HIR::HeaderStackType::nextIndexFieldName),
                getUIntConstant(loc, 1, 32));
            // TODO: Insert verify() call inside parser
            setValue(m, last);
        } else if (m->member == P4::IR::Type_Array::next) {
            auto parent = getValue(m->expr);
            auto array = P4HIR::StructExtractOp::create(builder, loc, parent,
                                                        P4HIR::HeaderStackType::dataFieldName);
            auto next = P4HIR::StructExtractOp::create(builder, loc, parent,
                                                       P4HIR::HeaderStackType::nextIndexFieldName);
            // TODO: Insert verify() call
            auto field = P4HIR::ArrayGetOp::create(builder, loc, array, next);
            setValue(m, field);
        } else if (m->member == P4::IR::Type_Array::last) {
            auto parent = getValue(m->expr);
            auto array = P4HIR::StructExtractOp::create(builder, loc, parent,
                                                        P4HIR::HeaderStackType::dataFieldName);
            auto last = P4HIR::BinOp::create(
                builder, loc, P4HIR::BinOpKind::Sub,
                P4HIR::StructExtractOp::create(builder, loc, parent,
                                               P4HIR::HeaderStackType::nextIndexFieldName),
                getUIntConstant(loc, 1, 32));
            // TODO: Insert verify() call
            auto field = P4HIR::ArrayGetOp::create(builder, loc, array, last);
            setValue(m, field);
        } else
            BUG("unknown header stack member %1% (aka %2%)", m, dbp(m));
    } else {
        BUG("cannot convert this member reference %1% (aka %2%) yet", m, dbp(m));
    }
}

bool P4HIRConverter::preorder(const P4::IR::StructExpression *str) {
    ConversionTracer trace("Converting ", str);

    auto type = getOrCreateType(str->structType);

    auto loc = getLoc(str);
    llvm::SmallVector<mlir::Value, 4> fields;
    for (const auto *field : str->components) {
        fields.push_back(convert(field->expression));
    }

    // If this is header, make it valid as well
    if (mlir::isa<P4HIR::HeaderType>(type))
        fields.push_back(emitValidityConstant(loc, P4HIR::ValidityBit::Valid));

    setValue(str, P4HIR::StructOp::create(builder, loc, type, fields).getResult());

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ListExpression *lst) {
    ConversionTracer trace("Converting ", lst);

    auto type = getOrCreateType(lst->type);

    auto loc = getLoc(lst);
    llvm::SmallVector<mlir::Value, 4> fields;
    for (const auto *field : lst->components) {
        fields.push_back(convert(field));
    }

    setValue(lst, P4HIR::TupleOp::create(builder, loc, type, fields).getResult());

    return false;
}

void P4HIRConverter::postorder(const P4::IR::ArrayIndex *arr) {
    ConversionTracer trace("Converting ", arr);

    auto lhs = getValue(arr->left);
    auto loc = getLoc(arr);
    if (mlir::isa<mlir::TupleType>(lhs.getType())) {
        auto idx = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(arr->right));
        setValue(arr, P4HIR::TupleExtractOp::create(builder, loc, lhs, idx).getResult());
        return;
    } else if (mlir::isa<P4HIR::ArrayType>(lhs.getType())) {
        auto idx = getValue(arr->right, getB32Type());
        setValue(arr, P4HIR::ArrayGetOp::create(builder, loc, lhs, idx).getResult());
        return;
    } else if (mlir::isa<P4HIR::HeaderStackType>(lhs.getType())) {
        auto idx = getValue(arr->right, getB32Type());
        auto dataField = P4HIR::StructExtractOp::create(builder, loc, lhs,
                                                        P4HIR::HeaderStackType::dataFieldName);
        setValue(arr, P4HIR::ArrayGetOp::create(builder, loc, dataField, idx).getResult());
        return;
    }

    BUG("cannot handle this array yet: %1%", arr);
}

void P4HIRConverter::postorder(const P4::IR::Range *range) {
    ConversionTracer trace("Converting ", range);

    auto lhs = getValue(range->left, getIntType(range->left->type));
    auto rhs = getValue(range->right, getIntType(range->right->type));

    auto loc = getLoc(range);
    setValue(range, P4HIR::RangeOp::create(builder, loc, lhs, rhs).getResult());
}

void P4HIRConverter::postorder(const P4::IR::Mask *range) {
    ConversionTracer trace("Converting ", range);

    auto lhs = getValue(range->left, getIntType(range->left->type));
    auto rhs = getValue(range->right, getIntType(range->right->type));

    auto loc = getLoc(range);
    setValue(range, P4HIR::MaskOp::create(builder, loc, lhs, rhs).getResult());
}

bool P4HIRConverter::preorder(const P4::IR::P4Parser *parser) {
    ConversionTracer trace("Converting ", parser);
    ValueScope scope(*p4Values);

    auto annotations = convert(parser->getAnnotations());

    auto applyType = mlir::cast<P4HIR::FuncType>(getOrCreateType(parser->getApplyMethodType()));
    auto ctorType =
        mlir::cast<P4HIR::CtorType>(getOrCreateConstructorType(parser->getConstructorMethodType()));
    auto argAttrs = convertParamAttributes(parser->getApplyParameters());
    assert(applyType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto loc = getLoc(parser);
    auto parserOp = P4HIR::ParserOp::create(builder, loc, parser->name.string_view(), applyType,
                                            ctorType, argAttrs, annotations);
    parserOp.createEntryBlock();
    auto parserSymbol = mlir::SymbolRefAttr::get(parserOp);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&parserOp.getBody().front());

    // Iterate over parameters again binding parameter values to arguments of first BB
    auto &body = parserOp.getBody();
    const auto &params = parser->getApplyParameters()->parameters;

    assert(body.getNumArguments() == params.size() && "invalid parameter conversion");
    for (auto [param, bodyArg] : llvm::zip(params, body.getArguments())) setValue(param, bodyArg);

    // Constructor arguments are special: they are compile-time constants,
    // create placeholders for them
    for (const auto *param : parser->getConstructorParameters()->parameters) {
        llvm::StringRef paramName = param->name.string_view();
        auto paramType = getOrCreateType(param);
        auto placeholder =
            P4HIR::CtorParamAttr::get(paramType, parserSymbol, builder.getStringAttr(paramName));
        auto val = P4HIR::ConstOp::create(builder, getLoc(param), placeholder, paramName);
        setValue(param, val);
    }

    {
        SymbolScope symbols(p4Symbols);

        // Materialize locals
        visit(parser->parserLocals);

        // Walk over all states, materializing the bodies
        visit(parser->states);

        // Create default transition (to start state)
        P4HIR::ParserTransitionOp::create(
            builder, getEndLoc(builder, parser),
            getQualifiedSymbolRef(P4::IR::ParserState::start.string_view()));
    }

    setSymbol(parser, parserSymbol);
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ParserState *state) {
    ConversionTracer trace("Converting ", state);
    ValueScope scope(*p4Values);

    auto annotations = convert(state->annotations);

    auto stateOp =
        P4HIR::ParserStateOp::create(builder, getLoc(state), state->name.string_view(),
                                     annotations.empty() ? mlir::DictionaryAttr() : annotations);
    mlir::Block &first = stateOp.getBody().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&first);

    // Materialize all state components
    visit(state->components);

    // accept / reject states are special, their bodies contain only accept / reject ops
    if (state->name == P4::IR::ParserState::accept) {
        P4HIR::ParserAcceptOp::create(builder, getLoc(state));
        visitAgain();
        return false;
    } else if (state->name == P4::IR::ParserState::reject) {
        P4HIR::ParserRejectOp::create(builder, getLoc(state));
        visitAgain();
        return false;
    }

    // Normal transition is either PathExpression or SelectExpression
    if (const auto *pe = state->selectExpression->to<P4::IR::PathExpression>()) {
        LOG4("Resolving direct transition: " << pe);
        auto loc = getLoc(pe);
        const auto *nextState = resolvePath(pe->path, false)->checkedTo<P4::IR::ParserState>();
        // next state might not exist yet, so we do not use p4symbols here and
        // build symbol reference by hand
        P4HIR::ParserTransitionOp::create(builder, loc,
                                          getQualifiedSymbolRef(nextState->name.string_view()));
    } else {
        LOG4("Resolving select transition: " << state->selectExpression);
        visit(state->selectExpression);
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::SelectExpression *select) {
    ConversionTracer trace("Converting ", select);

    // Materialize values to select over. Select is always a ListExpression,
    // even if it contains a single value. Unpack the top-level select tuple
    // to its individual components for p4hir.transition_select.
    const auto &comps = select->select->components;
    llvm::SmallVector<mlir::Value, 4> operands;
    for (const P4::IR::Node *comp : comps) operands.push_back(convert(comp));

    auto transitionSelectOp =
        P4HIR::ParserTransitionSelectOp::create(builder, getLoc(select), operands);
    mlir::Block &first = transitionSelectOp.getBody().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&first);

    bool hasDefaultCase = false;
    for (const auto *selectCase : select->selectCases) {
        const auto *nextState =
            resolvePath(selectCase->state->path, false)->checkedTo<P4::IR::ParserState>();
        P4HIR::ParserSelectCaseOp::create(
            builder, getLoc(selectCase),
            [&](mlir::OpBuilder &b, mlir::Location) {
                const auto *keyset = selectCase->keyset;
                auto endLoc = getEndLoc(builder, keyset);

                // Type inference does not do proper type unification for the key,
                // so we'd need to do this by ourselves
                auto convertElement = [&](const P4::IR::Expression *expr) -> mlir::Value {
                    // Universal set
                    if (expr->is<P4::IR::DefaultExpression>())
                        return getUniversalSetConstant(endLoc);

                    auto elVal = convert(expr);
                    if (!mlir::isa<P4HIR::SetType>(elVal.getType()))
                        elVal = P4HIR::SetOp::create(b, getEndLoc(builder, expr), elVal);
                    return elVal;
                };

                // Create a variadic yield expression from the keys in the keyset.
                llvm::SmallVector<mlir::Value, 4> elements;
                if (const auto *keyList = keyset->to<P4::IR::ListExpression>()) {
                    for (const auto *element : keyList->components)
                        elements.push_back(convertElement(element));
                } else {
                    elements.push_back(convertElement(keyset));
                }

                hasDefaultCase |= llvm::all_of(elements, P4HIR::isUniversalSetValue);
                P4HIR::YieldOp::create(b, endLoc, elements);
            },
            getQualifiedSymbolRef(nextState->name.string_view()));
    }

    // If there is no default case, then synthesize one explicitly
    // FIXME: signal `error.NoMatch` error.
    if (!hasDefaultCase) {
        auto endLoc = getEndLoc(builder, select);
        P4HIR::ParserSelectCaseOp::create(
            builder, endLoc,
            [&](mlir::OpBuilder &b, mlir::Location) {
                P4HIR::YieldOp::create(b, endLoc, getUniversalSetConstant(endLoc));
            },
            getQualifiedSymbolRef(P4::IR::ParserState::reject.string_view()));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Instance *decl) {
    ConversionTracer trace("Converting ", decl);

    auto annotations = convert(decl->annotations);

    // P4::Instantiation goes via typeMap and it returns some weird clone
    // instead of converted type
    const auto *type = resolveType(decl->type);
    CHECK_NULL(type);
    LOG4("Resolved to: " << dbp(type));

    llvm::SmallVector<mlir::Value, 4> operands;
    llvm::DenseMap<const P4::IR::Argument *, mlir::Value> argValues;
    for (const auto *arg : *decl->arguments) {
        ConversionTracer trace("Converting ", arg);
        argValues.try_emplace(arg, convert(arg->expression));
    }

    // Resolve to base type
    if (const auto *tdef = type->to<P4::IR::Type_Typedef>()) {
        type = resolveType(tdef->type);
        CHECK_NULL(type);
        LOG4("Resolved to typedef type: " << dbp(type));
    }
    llvm::SmallVector<mlir::Type> typeParameters;
    if (const auto *spec = type->to<P4::IR::Type_Specialized>()) {
        for (const auto *type : *spec->arguments) {
            typeParameters.push_back(getOrCreateType(type));
        }
        type = resolveType(spec->baseType);
        CHECK_NULL(type);
        LOG4("Resolved to base type: " << dbp(type));
    }

    // Shuffle arguments into proper order
    auto populateOperands = [&](const P4::IR::ParameterList *params) {
        P4::ParameterSubstitution subst;
        subst.populate(params, decl->arguments);
        for (const auto &param : params->parameters) {
            auto argument = subst.lookup(param);
            auto argVal = argValues.lookup(argument);
            // Create a placeholder for @optional arguments
            if (!argVal && param->isOptional())
                argVal =
                    P4HIR::UninitializedOp::create(builder, getLoc(decl), getOrCreateType(param));
            BUG_CHECK(argVal, "unconverted argument for parameter %1%", param);
            operands.push_back(argVal);
        }
    };

    if (const auto *cont = type->to<P4::IR::IContainer>()) {
        populateOperands(cont->getConstructorParameters());
    } else {
        const auto *ext = type->checkedTo<P4::IR::Type_Extern>();
        const auto *ctor = ext->lookupConstructor(decl->arguments);
        populateOperands(ctor->getParameters());
    }
    // TODO: Reduce code duplication below. Unify with ConstructCallExpression
    auto nameAttr = builder.getStringAttr(decl->name.string_view());
    if (const auto *parser = type->to<P4::IR::P4Parser>()) {
        LOG4("resolved as parser instantiation");
        auto parserSym = p4Symbols.lookup(parser);
        BUG_CHECK(parserSym, "expected reference parser to be converted: %1%", dbp(parser));

        auto instance = P4HIR::InstantiateOp::create(builder, getLoc(decl), parserSym, operands,
                                                     nameAttr, annotations);
        setSymbol(decl, getQualifiedSymbolRef(instance));
    } else if (const auto *ext = type->to<P4::IR::Type_Extern>()) {
        LOG4("resolved as extern instantiation");
        auto externName = builder.getStringAttr(ext->name.string_view());
        auto ctorName = mlir::SymbolRefAttr::get(builder.getContext(), externName);
        auto fullCtorName = mlir::SymbolRefAttr::get(builder.getContext(), externName, {ctorName});

        auto instance = P4HIR::InstantiateOp::create(builder, getLoc(decl), fullCtorName, operands,
                                                     nameAttr, typeParameters, annotations);
        setSymbol(decl, getQualifiedSymbolRef(instance));
    } else if (const auto *control = type->to<P4::IR::P4Control>()) {
        LOG4("resolved as control instantiation");
        auto controlSym = p4Symbols.lookup(control);
        BUG_CHECK(controlSym, "expected reference control to be converted: %1%", dbp(control));

        auto instance = P4HIR::InstantiateOp::create(builder, getLoc(decl), controlSym, operands,
                                                     nameAttr, annotations);
        setSymbol(decl, getQualifiedSymbolRef(instance));
    } else if (const auto *pkg = type->to<P4::IR::Type_Package>()) {
        LOG4("resolved as package instantiation");
        auto packageName = mlir::SymbolRefAttr::get(builder.getContext(), pkg->name.string_view());
        auto instance = P4HIR::InstantiateOp::create(builder, getLoc(decl), packageName, operands,
                                                     nameAttr, typeParameters, annotations);
        setSymbol(decl, getQualifiedSymbolRef(instance));
    } else {
        BUG("unsupported instance: %1% (of type %2%)", decl, dbp(type));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Type_Extern *ext) {
    auto loc = getLoc(ext);

    // TODO: Move to common method
    llvm::SmallVector<mlir::Type> typeParameters;
    for (const auto *type : ext->getTypeParameters()->parameters) {
        typeParameters.push_back(getOrCreateType(type));
    }

    auto annotations = convert(ext->annotations);

    auto extOp =
        P4HIR::ExternOp::create(builder, loc, ext->name.string_view(), typeParameters, annotations);
    extOp.createEntryBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&extOp.getBody().front());

    // Materialize method declarations
    visit(ext->methods);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Type_Package *pkg) {
    auto loc = getLoc(pkg);

    auto annotations = convert(pkg->annotations);

    // TODO: Move to common method
    llvm::SmallVector<mlir::Type> typeParameters;
    for (const auto *type : pkg->getTypeParameters()->parameters) {
        typeParameters.push_back(getOrCreateType(type));
    }

    auto ctorType =
        mlir::cast<P4HIR::CtorType>(getOrCreateConstructorType(pkg->getConstructorMethodType()));

    auto argAttrs = convertParamAttributes(pkg->getConstructorParameters());
    assert(ctorType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    mlir::OpBuilder::InsertionGuard guard(builder);

    // Check if there is a declaration with the same name in the current symbol table.
    // If yes, create / add to an overload set
    auto *parentOp = builder.getBlock()->getParentOp();
    auto origSymName = builder.getStringAttr(pkg->name.string_view());
    auto symName = origSymName;
    if (auto *otherOp = mlir::SymbolTable::lookupNearestSymbolFrom(parentOp, origSymName)) {
        LOG4("Package constructor is overloaded");

        P4HIR::OverloadSetOp ovl;
        auto getUniqueName = [&](mlir::StringAttr toRename) {
            unsigned counter = 0;
            return mlir::SymbolTable::generateSymbolName<256>(
                toRename,
                [&](llvm::StringRef candidate) {
                    return ovl.lookupSymbol(builder.getStringAttr(candidate)) != nullptr;
                },
                counter);
        };

        if (auto otherPkg = llvm::dyn_cast<P4HIR::PackageOp>(otherOp)) {
            LOG4("Creating overload set");

            ovl = P4HIR::OverloadSetOp::create(builder, loc, origSymName);
            builder.setInsertionPointToStart(&ovl.createEntryBlock());
            otherPkg->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());

            // Unique the symbol name to avoid clashes in the symbol table.  The
            // overload set takes over the symbol name. Still, all the symbols
            // in `p4Symbol` are created wrt the original name, so we do not use
            // SymbolTable::rename() here.
            otherPkg.setSymName(getUniqueName(origSymName));
        } else {
            LOG4("Adding to overload set");

            ovl = llvm::cast<P4HIR::OverloadSetOp>(otherOp);
            builder.setInsertionPointToEnd(&ovl.getBody().front());
        }

        symName = builder.getStringAttr(getUniqueName(symName));

        LOG4("Translated: " << origSymName.getValue().str() << " -> " << symName.getValue().str());
    }

    P4HIR::PackageOp::create(builder, loc, symName, ctorType, typeParameters, argAttrs,
                             annotations);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Control *control) {
    ConversionTracer trace("Converting ", control);
    ValueTable controlValues, *savedValues = p4Values;
    p4Values = &controlValues;
    ValueScope scope(controlValues);

    auto annotations = convert(control->getAnnotations());

    auto applyType = mlir::cast<P4HIR::FuncType>(getOrCreateType(control->getApplyMethodType()));
    auto ctorType = mlir::cast<P4HIR::CtorType>(
        getOrCreateConstructorType(control->getConstructorMethodType()));

    auto argAttrs = convertParamAttributes(control->getApplyParameters());
    assert(applyType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto loc = getLoc(control);
    auto controlOp = P4HIR::ControlOp::create(builder, loc, control->name.string_view(), applyType,
                                              ctorType, argAttrs, annotations);
    controlOp.createEntryBlock();
    auto controlSymbol = mlir::SymbolRefAttr::get(controlOp);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&controlOp.getBody().front());

    // Iterate over parameters again binding parameter values to arguments of first BB
    auto &body = controlOp.getBody();
    const auto &params = control->getApplyParameters()->parameters;

    assert(body.getNumArguments() == params.size() && "invalid parameter conversion");
    for (auto [param, bodyArg] : llvm::zip(params, body.getArguments())) setValue(param, bodyArg);

    // Constructor arguments are special: they are compile-time constants,
    // create placeholders for them
    for (const auto *param : control->getConstructorParameters()->parameters) {
        llvm::StringRef paramName = param->name.string_view();
        auto paramType = getOrCreateType(param);
        auto placeholder =
            P4HIR::CtorParamAttr::get(paramType, controlSymbol, builder.getStringAttr(paramName));
        auto val = P4HIR::ConstOp::create(builder, getLoc(param), placeholder, paramName);
        setValue(param, val);
    }

    // Materialize locals
    {
        SymbolScope symbols(p4Symbols);

        auto getUniqueName = [&](mlir::StringAttr toRename) {
            unsigned counter = 0;
            return mlir::SymbolTable::generateSymbolName<256>(
                toRename,
                [&](llvm::StringRef candidate) {
                    return controlOp.lookupSymbol(builder.getStringAttr(candidate)) != nullptr;
                },
                counter);
        };

        // Actions could refer to control's arguments. Materialize them as control locals
        for (auto [param, bodyArg] : llvm::zip(params, body.getArguments())) {
            auto nameAttr = getUniqueName(builder.getStringAttr(llvm::Twine("__local_") +
                                                                control->name.string_view() + "_" +
                                                                param->name.string_view()));
            auto local = P4HIR::ControlLocalOp::create(builder, getLoc(param), nameAttr, bodyArg);
            setSymbol(param, getQualifiedSymbolRef(local));
        }

        for (const auto *local : control->controlLocals) {
            visit(local);
            // Create symbols for variables. Instantiations are symbols by
            // themselves, no need to do something special
            if (const auto *var = local->to<P4::IR::Declaration_Variable>()) {
                auto val = p4Values->lookup(local);
                auto nameAttr = getUniqueName(builder.getStringAttr(llvm::Twine("__local_") +
                                                                    control->name.string_view() +
                                                                    "_" + var->name.string_view()));
                auto local = P4HIR::ControlLocalOp::create(builder, getLoc(var), nameAttr, val);
                setSymbol(var, getQualifiedSymbolRef(local));
            }
        }

        {
            ValueScope scope(*p4Values);

            auto applyOp = P4HIR::ControlApplyOp::create(builder, getLoc(control->body));
            mlir::Block &first = applyOp.getBody().emplaceBlock();

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&first);

            // Materialize body
            visit(control->body);
        }
    }

    setSymbol(control, controlSymbol);

    p4Values = savedValues;

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Table *table) {
    ConversionTracer trace("Converting ", table);

    auto annotations = convert(table->annotations);

    auto loc = getLoc(table);
    auto tableOp = P4HIR::TableOp::create(builder, loc, table->name.string_view(), annotations,
                                          [&](mlir::OpBuilder &, mlir::Location) {
                                              // Materialize all properties
                                              for (const auto *prop : table->properties->properties)
                                                  visit(prop);
                                          });

    setSymbol(table, getQualifiedSymbolRef(tableOp));
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ActionListElement *act) {
    ValueScope scope(controlPlaneValues);

    auto annotations = convert(act->annotations);

    // We expect that everything was normalized to action calls.
    const auto *expr = act->expression->checkedTo<P4::IR::MethodCallExpression>();
    // Prepare control plane values. These will be filled in visit() call
    const auto *actType = expr->method->type->checkedTo<P4::IR::Type_Action>();

    llvm::SmallVector<mlir::Type> controlPlaneTypes;
    llvm::SmallVector<mlir::DictionaryAttr> controlPlaneParamAttrs;
    size_t argCount = expr->arguments->size();
    const auto &params = actType->parameters->parameters;
    for (size_t idx = argCount; idx < params.size(); ++idx) {
        const auto *param = params[idx];
        controlPlaneTypes.push_back(getOrCreateType(param));
        auto pAnnotations = convert(param->annotations);

        llvm::SmallVector<mlir::NamedAttribute> paramAttrs = {
            builder.getNamedAttr(P4HIR::FuncOp::getParamNameAttrName(),
                                 builder.getStringAttr(param->name.string_view())),
        };
        if (!pAnnotations.empty())
            paramAttrs.emplace_back(
                builder.getNamedAttr(P4HIR::FuncOp::getParamAnnotationAttrName(), annotations));
        controlPlaneParamAttrs.emplace_back(builder.getDictionaryAttr(paramAttrs));
    }

    auto funcType = P4HIR::FuncType::get(context(), controlPlaneTypes);
    const auto *action = resolvePath(expr->method->checkedTo<P4::IR::PathExpression>()->path, false)
                             ->checkedTo<P4::IR::P4Action>();
    auto actSym = p4Symbols.lookup(action);
    BUG_CHECK(actSym, "expected reference action to be converted: %1%", action);

    auto localActSym = mlir::SymbolRefAttr::get(actSym.getLeafReference());

    P4HIR::TableActionOp::create(
        builder, getLoc(expr), localActSym, funcType, controlPlaneParamAttrs, annotations,
        [&](mlir::OpBuilder &, mlir::Block::BlockArgListType args, mlir::Location) {
            for (const auto arg : args) {
                const auto *param = params[argCount + arg.getArgNumber()];
                controlPlaneValues.insert(param, arg);
            }

            visit(expr);
        });

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Entry *ent) {
    ValueScope scope(*p4Values);

    auto annotations = convert(ent->annotations);
    mlir::TypedAttr priority;
    if (ent->priority) priority = getOrCreateConstantExpr(ent->priority);
    auto keys = getOrCreateConstantExpr(ent->keys);

    // We expect that everything was normalized to action calls.
    const auto *expr = ent->action->checkedTo<P4::IR::MethodCallExpression>();

    P4HIR::TableEntryOp::create(builder, getLoc(ent), keys, ent->isConst, priority, annotations,
                                [&](mlir::OpBuilder &, mlir::Location) { visit(expr); });

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Property *prop) {
    ConversionTracer trace("Converting ", prop);

    auto annotations = convert(prop->annotations);

    auto loc = getLoc(prop);
    if (0) {
    } else if (prop->name == P4::IR::TableProperties::actionsPropertyName) {
        P4HIR::TableActionsOp::create(
            builder, loc, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                const auto *alist = prop->value->checkedTo<P4::IR::ActionList>();
                for (const auto *act : alist->actionList) visit(act);
            });
    } else if (prop->name == P4::IR::TableProperties::keyPropertyName) {
        auto emptyFuncType = P4HIR::FuncType::get(builder.getContext(), {});
        auto tableKeyOp = P4HIR::TableKeyOp::create(
            builder, loc, emptyFuncType, llvm::SmallVector<mlir::DictionaryAttr>{}, annotations);
        tableKeyOp.createEntryBlock();
        mlir::Block *tableKeyBlock = &tableKeyOp.getBody().front();

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(tableKeyBlock);

        ValueScope scope(*p4Values);

        const auto *key = prop->value->checkedTo<P4::IR::Key>();
        for (const auto *kel : key->keyElements) {
            auto kAnnotations = convert(kel->annotations);
            auto kExpr = convert(kel->expression);
            const auto *match_kind =
                resolvePath(kel->matchType->path, false)->checkedTo<P4::IR::Declaration_ID>();
            P4HIR::TableKeyEntryOp::create(builder, getLoc(kel),
                                           builder.getStringAttr(match_kind->name.string_view()),
                                           kExpr, kAnnotations);
        }

        // Create new block arguments on-demand for values coming from an outer scope.
        llvm::MapVector<mlir::Value, mlir::Value> argMapping;
        auto adjustVal = [&](mlir::Value val) {
            bool outerVal =
                !val.getDefiningOp() || !tableKeyBlock->findAncestorOpInBlock(*val.getDefiningOp());
            if (!outerVal) return val;

            auto [it, ins] = argMapping.insert({val, {}});
            if (ins) it->second = tableKeyBlock->addArgument(val.getType(), loc);

            return it->second;
        };

        tableKeyBlock->walk([&](mlir::Operation *op) {
            for (mlir::OpOperand &operand : op->getOpOperands())
                operand.assign(adjustVal(operand.get()));
        });

        // Update function type based on the used outer values.
        auto inputTypes = llvm::map_to_vector(tableKeyBlock->getArguments(),
                                              [](auto arg) { return arg.getType(); });
        auto funcType = P4HIR::FuncType::get(builder.getContext(), inputTypes);
        tableKeyOp.setFunctionTypeAttr(mlir::TypeAttr::get(funcType));

        // Store the original values used in order to build the table apply call.
        const P4::IR::P4Table *table = getParent<P4::IR::P4Table>();
        assert(table && "Expected to find outer table for table key");
        auto tableKeyArgs = llvm::map_to_vector(argMapping, [](const auto &p) { return p.first; });
        [[maybe_unused]] auto [it, ins] = tableKeyArgsMap.insert({table, tableKeyArgs});
        assert(ins && "Expected unique table key");
    } else if (prop->name == P4::IR::TableProperties::defaultActionPropertyName) {
        P4HIR::TableDefaultActionOp::create(
            builder, loc, prop->isConstant, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(*p4Values);

                const auto *expr = prop->value->checkedTo<P4::IR::ExpressionValue>()->expression;
                visit(expr);
            });
    } else if (prop->name == P4::IR::TableProperties::entriesPropertyName) {
        P4HIR::TableEntriesOp::create(
            builder, loc, prop->isConstant, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                const auto *elist = prop->value->checkedTo<P4::IR::EntriesList>();
                for (const auto *entry : elist->entries) visit(entry);
            });
    } else if (prop->name == P4::IR::TableProperties::sizePropertyName) {
        const auto *expr = prop->value->checkedTo<P4::IR::ExpressionValue>()->expression;
        // Here property value might be a constructor argument. So we need to
        // see, if we have a placeholder for it
        mlir::TypedAttr size;
        if (auto val = getValue(expr, {}, /* unchecked */ true))
            size = mlir::cast<P4HIR::ConstOp>(val.getDefiningOp()).getValue();
        else
            size = getOrCreateConstantExpr(expr);

        P4HIR::TableSizeOp::create(builder, loc, size,
                                   annotations.empty() ? mlir::DictionaryAttr() : annotations);
    } else {
        P4HIR::TablePropertyOp::create(
            builder, loc, builder.getStringAttr(prop->getName().string_view()), prop->isConstant,
            annotations, [&](mlir::OpBuilder &b, mlir::Type &resultType, mlir::Location) {
                ValueScope scope(*p4Values);

                mlir::Value val;
                if (const auto *exprVal = prop->value->to<P4::IR::ExpressionValue>()) {
                    val = convert(exprVal->expression);
                } else {
                    const auto *exprLVal = prop->value->checkedTo<P4::IR::ExpressionListValue>();
                    llvm::SmallVector<mlir::Value> vals;
                    llvm::SmallVector<mlir::Type> types;
                    for (const auto *expr : exprLVal->expressions) {
                        vals.push_back(convert(expr));
                        types.push_back(vals.back().getType());
                    }
                    val =
                        P4HIR::TupleOp::create(b, getEndLoc(b, prop), b.getTupleType(types), vals);
                }
                resultType = val.getType();
                P4HIR::YieldOp::create(b, getEndLoc(b, prop), val);
            });
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::SwitchStatement *sw) {
    ConversionTracer trace("Converting ", sw);

    auto cond = convert(sw->expression);

    P4HIR::SwitchOp::create(builder, getLoc(sw), cond, [&](mlir::OpBuilder &b, mlir::Location) {
        llvm::SmallVector<mlir::Attribute> cases;

        for (const auto *swCase : sw->cases) {
            if (!swCase->label->to<P4::IR::DefaultExpression>()) {
                // Handle special case: action run enum
                if (sw->expression->type->is<P4::IR::Type_ActionEnum>()) {
                    const auto *path = swCase->label->checkedTo<P4::IR::PathExpression>();
                    cases.push_back(
                        P4HIR::EnumFieldAttr::get(cond.getType(), path->path->name.string_view()));
                } else
                    cases.push_back(getOrCreateConstantExpr(swCase->label));
            }

            if (swCase->statement) {
                P4HIR::CaseOpKind caseOpKind;
                if (swCase->label->to<P4::IR::DefaultExpression>())
                    caseOpKind = P4HIR::CaseOpKind::Default;
                else
                    caseOpKind =
                        cases.size() > 1 ? P4HIR::CaseOpKind::Anyof : P4HIR::CaseOpKind::Equal;

                P4HIR::CaseOp::create(builder, getLoc(swCase), b.getArrayAttr(cases), caseOpKind,
                                      [&](mlir::OpBuilder &b, mlir::Location) {
                                          ValueScope scope(*p4Values);

                                          visit(swCase->statement);
                                          P4HIR::YieldOp::create(b, getEndLoc(builder, swCase));
                                      });
                cases.clear();
            }
        }

        P4HIR::YieldOp::create(b, getEndLoc(builder, sw));
    });

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ForStatement *fstmt) {
    ConversionTracer trace("Converting ", fstmt);

    auto annotations = convert(fstmt->annotations);
    mlir::OpBuilder::InsertionGuard guard(builder);

    // We only wrap our for loop within a dedicated block scope when there are any
    // declarations in our init statements to limit the lifetimes of loop-local variables.
    bool emitScope =
        std::any_of(fstmt->init.begin(), fstmt->init.end(),
                    [](const P4::IR::StatOrDecl *stmt) { return stmt->is<P4::IR::Declaration>(); });

    auto buildForLoop = [&](mlir::OpBuilder &b, mlir::Location loc) {
        ValueScope scope(*p4Values);

        visit(fstmt->init);

        P4HIR::ForOp::create(
            b, loc, annotations,
            /*condBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(*p4Values);

                auto cond = convert(fstmt->condition);
                P4HIR::ConditionOp::create(b, getEndLoc(builder, fstmt->condition), cond);
            },
            /*bodyBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(*p4Values);

                visit(fstmt->body);
                P4HIR::buildTerminatedBody(b, getEndLoc(builder, fstmt->body));
            },
            /*updatesBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(*p4Values);

                visit(fstmt->updates);
                const auto *locNode = fstmt->updates.empty() ? fstmt : fstmt->updates.back();
                P4HIR::buildTerminatedBody(b, getEndLoc(builder, locNode));
            });
    };

    if (emitScope) {
        auto scope = P4HIR::ScopeOp::create(builder, getLoc(fstmt), buildForLoop);
        builder.setInsertionPointToEnd(&scope.getScopeRegion().back());
        P4HIR::buildTerminatedBody(builder, getEndLoc(builder, fstmt));
    } else {
        buildForLoop(builder, getLoc(fstmt));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ForInStatement *forin) {
    ConversionTracer trace("Converting ", forin);

    auto annotations = convert(forin->annotations);
    mlir::OpBuilder::InsertionGuard guard(builder);

    bool emitScope = forin->collection->is<P4::IR::Range>();

    auto buildForInLoop = [&](mlir::OpBuilder &b, mlir::Location loc) {
        ValueScope scope(*p4Values);

        auto collection = convert(forin->collection);

        P4HIR::ForInOp::create(b, loc, collection, annotations,
                               /*bodyBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Value iterationArg, mlir::Location) {
                                   ValueScope scope(*p4Values);

                                   setValue(forin->decl, iterationArg);

                                   visit(forin->body);
                                   P4HIR::buildTerminatedBody(b, getEndLoc(builder, forin->body));
                               });
    };

    if (emitScope) {
        auto scope = P4HIR::ScopeOp::create(builder, getLoc(forin), buildForInLoop);
        builder.setInsertionPointToEnd(&scope.getScopeRegion().back());
        P4HIR::buildTerminatedBody(builder, getEndLoc(builder, forin));
    } else {
        buildForInLoop(builder, getLoc(forin));
    }

    return false;
}

mlir::OwningOpRef<mlir::ModuleOp> P4::P4MLIR::toMLIR(mlir::MLIRContext &context,
                                                     const P4::IR::P4Program *program,
                                                     P4::TypeMap *typeMap) {
    mlir::OpBuilder builder(&context);
    P4HIRConverter conv(builder, typeMap, true);

    auto moduleOp = mlir::ModuleOp::create(conv.getLoc(program));
    builder.setInsertionPointToEnd(moduleOp.getBody());

    if (auto sourceInfo = program->getSourceInfo(); sourceInfo.isValid()) {
        moduleOp.setSymName(sourceInfo.getSourceFile().string_view());
        moduleOp->setLoc(conv.getLoc(program));
    }
    program->apply(conv);

    if (!program || P4::errorCount() > 0) return nullptr;

    if (failed(mlir::verify(moduleOp))) {
        // Dump for debugging purposes
        moduleOp->print(llvm::outs());
        moduleOp.emitError("module verification error");
        return nullptr;
    }

    return moduleOp;
}

mlir::OwningOpRef<mlir::ModuleOp> P4::P4MLIR::toMLIR(P4HIRConverter &conv,
                                                     const P4::IR::P4Program *program) {
    mlir::OpBuilder &builder = conv.getBuilder();

    auto moduleOp = mlir::ModuleOp::create(conv.getLoc(program));
    builder.setInsertionPointToEnd(moduleOp.getBody());

    if (auto sourceInfo = program->getSourceInfo(); sourceInfo.isValid()) {
        moduleOp.setSymName(sourceInfo.getSourceFile().string_view());
        moduleOp->setLoc(conv.getLoc(program));
    }
    program->apply(conv);

    if (!program || P4::errorCount() > 0) return nullptr;

    if (failed(mlir::verify(moduleOp))) {
        // Dump for debugging purposes
        moduleOp->print(llvm::outs());
        moduleOp.emitError("module verification error");
        return nullptr;
    }

    return moduleOp;
}
