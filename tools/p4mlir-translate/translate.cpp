#include "translate.h"

#include <algorithm>
#include <climits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/methodInstance.h"
#include "frontends/p4/typeMap.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/big_int.h"
#include "lib/indent.h"
#include "lib/log.h"
#pragma GCC diagnostic pop

#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

using namespace P4::P4MLIR;

namespace {

// Converts P4 SourceLocation stored in 'node' into its MLIR counterpart
mlir::Location getLoc(mlir::OpBuilder &builder, const P4::IR::Node *node) {
    CHECK_NULL(node);
    auto sourceInfo = node->getSourceInfo();
    if (!sourceInfo.isValid()) return mlir::UnknownLoc::get(builder.getContext());

    const auto &start = sourceInfo.getStart();

    return mlir::FileLineColLoc::get(
        builder.getStringAttr(sourceInfo.getSourceFile().string_view()), start.getLineNumber(),
        start.getColumnNumber());
}

mlir::Location getEndLoc(mlir::OpBuilder &builder, const P4::IR::Node *node) {
    CHECK_NULL(node);
    auto sourceInfo = node->getSourceInfo();
    if (!sourceInfo.isValid()) return mlir::UnknownLoc::get(builder.getContext());

    const auto &end = sourceInfo.getEnd();

    return mlir::FileLineColLoc::get(
        builder.getStringAttr(sourceInfo.getSourceFile().string_view()), end.getLineNumber(),
        end.getColumnNumber());
}

mlir::APInt toAPInt(const P4::big_int &value, unsigned bitWidth = 0) {
    std::vector<uint64_t> valueBits;
    // Export absolute value into 64-bit unsigned values, most significant bit last
    export_bits(value, std::back_inserter(valueBits), 64, false);

    if (!bitWidth) bitWidth = valueBits.size() * sizeof(valueBits[0]) * CHAR_BIT;

    mlir::APInt apValue(bitWidth, valueBits);
    if (value < 0) apValue.negate();

    return apValue;
}

class P4HIRConverter;
class P4TypeConverter;

class ConversionTracer {
 public:
    ConversionTracer(const char *Kind, const P4::IR::Node *node) {
        // TODO: Add TimeTrace here
        LOG4(P4::IndentCtl::indent << Kind << dbp(node) << (LOGGING(5) ? ":" : ""));
        LOG5(node);
    }
    ~ConversionTracer() { LOG4_UNINDENT; }
};

// A dedicated converter for conversion of the P4 types to their destination
// representation.
class P4TypeConverter : public P4::Inspector {
 public:
    P4TypeConverter(P4HIRConverter &converter) : converter(converter) {}

    profile_t init_apply(const P4::IR::Node *node) override {
        BUG_CHECK(!type, "Type already converted");
        return Inspector::init_apply(node);
    }

    void end_apply(const P4::IR::Node *) override { BUG_CHECK(type, "Type not converted"); }

    bool preorder(const P4::IR::Node *node) override {
        BUG_CHECK(node->is<P4::IR::Type>(), "Invalid node");
        return false;
    }

    bool preorder(const P4::IR::Type *type) override {
        ::P4::error("%1%: P4 type not yet supported.", dbp(type));
        return false;
    }

    bool preorder(const P4::IR::Type_Bits *type) override;
    bool preorder(const P4::IR::Type_InfInt *type) override;
    bool preorder(const P4::IR::Type_Boolean *type) override;
    bool preorder(const P4::IR::Type_Unknown *type) override;
    bool preorder(const P4::IR::Type_Typedef *type) override {
        ConversionTracer trace("TypeConverting ", type);
        visit(type->type);
        return false;
    }

    bool preorder(const P4::IR::Type_Name *name) override;
    bool preorder(const P4::IR::Type_Action *act) override;
    bool preorder(const P4::IR::Type_Method *m) override;
    bool preorder(const P4::IR::Type_Void *v) override;
    bool preorder(const P4::IR::Type_Struct *s) override;
    bool preorder(const P4::IR::Type_Enum *e) override;
    bool preorder(const P4::IR::Type_SerEnum *se) override;

    mlir::Type getType() const { return type; }
    bool setType(const P4::IR::Type *type, mlir::Type mlirType);
    mlir::Type convert(const P4::IR::Type *type);

 private:
    P4HIRConverter &converter;
    mlir::Type type = nullptr;
};

class P4HIRConverter : public P4::Inspector, public P4::ResolutionContext {
    mlir::OpBuilder &builder;

    P4::TypeMap *typeMap = nullptr;
    llvm::DenseMap<const P4::IR::Type *, mlir::Type> p4Types;
    // TODO: Implement unified constant map
    // using CTVOrExpr = std::variant<const P4::IR::CompileTimeValue *,
    //                                const P4::IR::Expression *>;
    // llvm::DenseMap<CTVOrExpr, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Expression *, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Node *, mlir::Value> p4Values;
    using P4Symbol =
        std::variant<const P4::IR::P4Action *, const P4::IR::Function *, const P4::IR::Method *>;
    // TODO: Implement better scoped symbol table
    llvm::DenseMap<P4Symbol, mlir::SymbolRefAttr> p4Symbols;

    mlir::TypedAttr resolveConstant(const P4::IR::CompileTimeValue *ctv);
    mlir::Value resolveReference(const P4::IR::Node *node, bool unchecked = true);

    mlir::Value getBoolConstant(mlir::Location loc, bool value) {
        auto boolType = P4HIR::BoolType::get(context());
        return builder.create<P4HIR::ConstOp>(loc,
                                              P4HIR::BoolAttr::get(context(), boolType, value));
    }

 public:
    P4HIRConverter(mlir::OpBuilder &builder, P4::TypeMap *typeMap)
        : builder(builder), typeMap(typeMap) {
        CHECK_NULL(typeMap);
    }

    mlir::Type findType(const P4::IR::Type *type) const { return p4Types.lookup(type); }

    void setType(const P4::IR::Type *type, mlir::Type mlirType) {
        auto [it, inserted] = p4Types.try_emplace(type, mlirType);
        BUG_CHECK(inserted, "duplicate conversion for %1%", type);
    }

    mlir::Type getOrCreateType(const P4::IR::Type *type) {
        P4TypeConverter cvt(*this);
        type->apply(cvt);
        return cvt.getType();
    }

    mlir::Type getOrCreateType(const P4::IR::Expression *expr) {
        return getOrCreateType(typeMap->getType(expr, true));
    }

    mlir::Type getOrCreateType(const P4::IR::Declaration_Variable *decl) {
        auto declType = getOrCreateType(decl->type);
        return P4HIR::ReferenceType::get(builder.getContext(), declType);
    }

    mlir::Value materializeConstantExpr(const P4::IR::Expression *expr);

    // TODO: Implement proper CompileTimeValue support
    /*
    mlir::TypedAttr setConstant(const P4::IR::CompileTimeValue *ctv, mlir::TypedAttr attr) {
        auto [it, inserted] = p4Constants.try_emplace(ctv, attr);
        BUG_CHECK(inserted, "duplicate conversion of %1%", ctv);
        return it->second;
    }
    */

    mlir::TypedAttr setConstantExpr(const P4::IR::Expression *expr, mlir::TypedAttr attr) {
        auto [it, inserted] = p4Constants.try_emplace(expr, attr);
        BUG_CHECK(inserted, "duplicate conversion of %1%", expr);
        return it->second;
    }

    // TODO: Implement proper CompileTimeValue support
    /*
    mlir::TypedAttr getOrCreateConstant(const P4::IR::CompileTimeValue *ctv) {
        BUG_CHECK(!ctv->is<P4::IR::Expression>(), "use getOrCreateConstantExpr() instead");
        auto cst = p4Constants.lookup(ctv);
        if (cst) return cst;

        cst = resolveConstant(ctv);

        BUG_CHECK(cst, "expected %1% to be converted as constant", ctv);
        return cst;
    }
    */

    mlir::TypedAttr getOrCreateConstantExpr(const P4::IR::Expression *expr);

    mlir::Value getValue(const P4::IR::Node *node) {
        // If this is a PathExpression, resolve it
        if (const auto *pe = node->to<P4::IR::PathExpression>()) {
            node = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
        }

        auto val = p4Values.lookup(node);
        BUG_CHECK(val, "expected %1% (aka %2%) to be converted", node, dbp(node));

        // See, if node is a top-level constant. If yes, then clone the value
        // into the present scope as other top-level things are
        // IsolatedFromAbove.
        // TODO: Save new constant value into scoped value tables when we will have one
        if (auto constOp = val.getDefiningOp<P4HIR::ConstOp>();
            constOp && mlir::isa_and_nonnull<mlir::ModuleOp>(constOp->getParentOp())) {
            val = builder.clone(*constOp)->getResult(0);
        }

        if (mlir::isa<P4HIR::ReferenceType>(val.getType()))
            // Getting value out of variable involves a load.
            return builder.create<P4HIR::ReadOp>(getLoc(builder, node), val);

        return val;
    }

    mlir::Value setValue(const P4::IR::Node *node, mlir::Value value) {
        if (!value) return value;

        if (LOGGING(4)) {
            std::string s;
            llvm::raw_string_ostream os(s);
            value.print(os, mlir::OpPrintingFlags().assumeVerified());
            LOG4("Converted " << dbp(node) << " -> \"" << s << "\"");
        }

        auto [it, inserted] = p4Values.try_emplace(node, value);
        BUG_CHECK(inserted, "duplicate conversion of %1%", node);
        return it->second;
    }

    mlir::MLIRContext *context() const { return builder.getContext(); }

    bool preorder(const P4::IR::Node *node) override {
        ::P4::error("P4 construct not yet supported: %1% (aka %2%)", node, dbp(node));
        return false;
    }

    bool preorder(const P4::IR::Type *type) override {
        ConversionTracer trace("Converting ", type);
        P4TypeConverter cvt(*this);
        type->apply(cvt);
        return false;
    }

    bool preorder(const P4::IR::P4Program *) override { return true; }
    bool preorder(const P4::IR::P4Action *a) override;
    bool preorder(const P4::IR::Function *f) override;
    bool preorder(const P4::IR::Method *m) override;
    bool preorder(const P4::IR::BlockStatement *block) override {
        // If this is a top-level block where scope is implied (e.g. function,
        // action, certain statements) do not create explicit scope.
        if (getParent<P4::IR::BlockStatement>()) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto scope = builder.create<P4HIR::ScopeOp>(
                getLoc(builder, block),                   /*scopeBuilder=*/
                [&](mlir::OpBuilder &, mlir::Location) {  // nothing is being yielded
                    visit(block->components);
                });
            builder.setInsertionPointToEnd(&scope.getScopeRegion().back());
            builder.create<P4HIR::YieldOp>(getEndLoc(builder, block));
        } else
            visit(block->components);
        return false;
    }

    bool preorder(const P4::IR::Constant *c) override {
        materializeConstantExpr(c);
        return false;
    }
    bool preorder(const P4::IR::BoolLiteral *b) override {
        materializeConstantExpr(b);
        return false;
    }
    bool preorder(const P4::IR::PathExpression *e) override {
        // Should be resolved eslewhere
        return false;
    }

#define HANDLE_IN_POSTORDER(NodeTy)                                 \
    bool preorder(const P4::IR::NodeTy *) override { return true; } \
    void postorder(const P4::IR::NodeTy *) override;

    // Unary ops
    HANDLE_IN_POSTORDER(Neg)
    HANDLE_IN_POSTORDER(LNot)
    HANDLE_IN_POSTORDER(UPlus)
    HANDLE_IN_POSTORDER(Cmpl)

    // Binary ops
    HANDLE_IN_POSTORDER(Mul)
    HANDLE_IN_POSTORDER(Div)
    HANDLE_IN_POSTORDER(Mod)
    HANDLE_IN_POSTORDER(Add)
    HANDLE_IN_POSTORDER(Sub)
    HANDLE_IN_POSTORDER(AddSat)
    HANDLE_IN_POSTORDER(SubSat)
    HANDLE_IN_POSTORDER(BOr)
    HANDLE_IN_POSTORDER(BAnd)
    HANDLE_IN_POSTORDER(BXor)

    // Concat
    HANDLE_IN_POSTORDER(Concat)

    // Comparisons
    HANDLE_IN_POSTORDER(Equ)
    HANDLE_IN_POSTORDER(Neq)
    HANDLE_IN_POSTORDER(Leq)
    HANDLE_IN_POSTORDER(Lss)
    HANDLE_IN_POSTORDER(Grt)
    HANDLE_IN_POSTORDER(Geq)

    HANDLE_IN_POSTORDER(Cast)
    HANDLE_IN_POSTORDER(Declaration_Variable)
    HANDLE_IN_POSTORDER(ReturnStatement)

#undef HANDLE_IN_POSTORDER

    void postorder(const P4::IR::Member *m) override;

    bool preorder(const P4::IR::Declaration_Constant *decl) override;
    bool preorder(const P4::IR::AssignmentStatement *assign) override;
    bool preorder(const P4::IR::Mux *mux) override;
    bool preorder(const P4::IR::LOr *lor) override;
    bool preorder(const P4::IR::LAnd *land) override;
    bool preorder(const P4::IR::IfStatement *ifs) override;
    bool preorder(const P4::IR::MethodCallStatement *) override {
        // We handle MethodCallExpression instead
        return true;
    }

    bool preorder(const P4::IR::MethodCallExpression *mce) override;
    bool preorder(const P4::IR::StructExpression *str) override;
    bool preorder(const P4::IR::Member *m) override;

    mlir::Value emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind);
    mlir::Value emitBinOp(const P4::IR::Operation_Binary *binop, P4HIR::BinOpKind kind);
    mlir::Value emitConcatOp(const P4::IR::Concat *concatop);
    mlir::Value emitCmp(const P4::IR::Operation_Relation *relop, P4HIR::CmpOpKind kind);
};

bool P4TypeConverter::preorder(const P4::IR::Type_Bits *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::BitsType::get(converter.context(), type->width_bits(), type->isSigned);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_InfInt *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::InfIntType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Boolean *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::BoolType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Unknown *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::UnknownType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Name *name) {
    if ((this->type = converter.findType(name))) return false;

    ConversionTracer trace("Resolving type by name ", name);
    const auto *type = converter.resolveType(name);
    CHECK_NULL(type);
    mlir::Type mlirType = convert(type);
    return setType(name, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Action *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Type, 4> argTypes;

    BUG_CHECK(type->returnType == nullptr, "actions should not have return type set");
    CHECK_NULL(type->parameters);

    for (const auto *p : type->parameters->parameters) {
        mlir::Type type = convert(p->type);
        argTypes.push_back(p->hasOut() ? P4HIR::ReferenceType::get(type) : type);
    }

    auto mlirType = P4HIR::FuncType::get(converter.context(), argTypes);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Method *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Type, 4> argTypes;

    CHECK_NULL(type->parameters);
    CHECK_NULL(type->returnType);

    mlir::Type resultType = convert(type->returnType);

    for (const auto *p : type->parameters->parameters) {
        mlir::Type type = convert(p->type);
        argTypes.push_back(p->hasOut() ? P4HIR::ReferenceType::get(type) : type);
    }

    auto mlirType = P4HIR::FuncType::get(argTypes, resultType);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Void *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::VoidType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Struct *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<P4HIR::StructType::FieldInfo, 4> fields;
    for (const auto *field : type->fields) {
        fields.push_back({mlir::StringAttr::get(converter.context(), field->name.string_view()),
                          convert(field->type)});
    }

    auto mlirType = P4HIR::StructType::get(converter.context(), type->name.string_view(), fields);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Enum *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Attribute, 4> cases;
    for (const auto *field : type->members) {
        cases.push_back(mlir::StringAttr::get(converter.context(), field->name.string_view()));
    }
    auto mlirType = P4HIR::EnumType::get(converter.context(), type->name.string_view(),
                                         mlir::ArrayAttr::get(converter.context(), cases));
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_SerEnum *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::NamedAttribute, 4> cases;

    auto enumType = mlir::cast<P4HIR::BitsType>(convert(type->type));
    for (const auto *field : type->members) {
        auto value = mlir::cast<P4HIR::IntAttr>(converter.getOrCreateConstantExpr(field->value));
        cases.emplace_back(mlir::StringAttr::get(converter.context(), field->name.string_view()),
                           value);
    }

    auto mlirType = P4HIR::SerEnumType::get(type->name.string_view(), enumType, cases);
    return setType(type, mlirType);
}

bool P4TypeConverter::setType(const P4::IR::Type *type, mlir::Type mlirType) {
    this->type = mlirType;
    converter.setType(type, mlirType);
    return false;
}

mlir::Type P4TypeConverter::convert(const P4::IR::Type *type) {
    if ((this->type = converter.findType(type))) return getType();

    visit(type);
    return getType();
}

// Resolve an l-value-kind expression, building access operation for each "layer".
mlir::Value P4HIRConverter::resolveReference(const P4::IR::Node *node, bool unchecked) {
    auto ref = p4Values.lookup(node);
    if (ref) return ref;

    ConversionTracer trace("Resolving reference ", node);
    // Check if this is a reference to a member of something we can recognize
    if (const auto *m = node->to<P4::IR::Member>()) {
        auto base = resolveReference(m->expr);
        auto field = builder.create<P4HIR::StructExtractRefOp>(getLoc(builder, m), base,
                                                               m->member.string_view());
        return setValue(m, field.getResult());
    }

    // If this is a PathExpression, resolve it to the actual declaration, usualy this
    // is a "leaf" case.
    if (const auto *pe = node->to<P4::IR::PathExpression>()) {
        node = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
    }

    ref = p4Values.lookup(node);
    if (!ref) {
        visit(node);
        ref = p4Values.lookup(node);
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
    if (const auto *cast = expr->to<P4::IR::Cast>()) {
        mlir::Type destType = getOrCreateType(cast);
        mlir::Type srcType = getOrCreateType(cast->expr);
        // Fold equal-type casts (e.g. due to typedefs)
        if (destType == srcType) return setConstantExpr(expr, getOrCreateConstantExpr(cast->expr));

        // Fold sign conversions
        if (auto destBitsType = mlir::dyn_cast<P4HIR::BitsType>(destType)) {
            if (auto srcBitsType = mlir::dyn_cast<P4HIR::BitsType>(srcType)) {
                assert(destBitsType.getWidth() == srcBitsType.getWidth() &&
                       "expected signess conversion only");
                auto castee = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(cast->expr));
                return setConstantExpr(
                    expr, P4HIR::IntAttr::get(context(), destBitsType, castee.getValue()));
            }
        }
    }
    if (const auto *str = expr->to<P4::IR::StructExpression>()) {
        auto type = getOrCreateType(str->type);
        llvm::SmallVector<mlir::Attribute, 4> fields;
        for (const auto *field : str->components)
            fields.push_back(getOrCreateConstantExpr(field->expression));
        return setConstantExpr(expr, P4HIR::AggAttr::get(type, builder.getArrayAttr(fields)));
    }
    if (const auto *m = expr->to<P4::IR::Member>()) {
        if (const auto *typeNameExpr = m->expr->to<P4::IR::TypeNameExpression>()) {
            auto baseType = mlir::cast<P4HIR::EnumType>(getOrCreateType(typeNameExpr->typeName));
            return setConstantExpr(expr,
                                   P4HIR::EnumFieldAttr::get(baseType, m->member.string_view()));
        }

        auto base = mlir::cast<P4HIR::AggAttr>(getOrCreateConstantExpr(m->expr));
        auto structType = mlir::cast<P4HIR::StructType>(base.getType());

        if (auto maybeIdx = structType.getFieldIndex(m->member.string_view())) {
            auto field = base.getFields()[*maybeIdx];
            auto fieldType = structType.getFieldType(m->member.string_view());

            // TODO: We'd likely would want to convert this to some kind of interface,
            if (mlir::isa<P4HIR::BoolType>(fieldType))
                return setConstantExpr(expr, mlir::cast<P4HIR::BoolAttr>(field));

            if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(fieldType))
                return setConstantExpr(expr, mlir::cast<P4HIR::IntAttr>(field));

            return setConstantExpr(expr, mlir::cast<P4HIR::AggAttr>(field));
        } else
            BUG("invalid member reference %1%", m);
    }

    BUG("cannot resolve this constant expression yet %1% (aka %2%)", expr, dbp(expr));
}

mlir::Value P4HIRConverter::materializeConstantExpr(const P4::IR::Expression *expr) {
    ConversionTracer trace("Materializing constant expression ", expr);

    auto init = getOrCreateConstantExpr(expr);
    auto loc = getLoc(builder, expr);

    auto val = builder.create<P4HIR::ConstOp>(loc, init);
    return setValue(expr, val);
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Constant *decl) {
    ConversionTracer trace("Converting ", decl);

    auto init = getOrCreateConstantExpr(decl->initializer);
    auto loc = getLoc(builder, decl);

    auto val = builder.create<P4HIR::ConstOp>(loc, init, decl->name.string_view());
    setValue(decl, val);

    return false;
}

void P4HIRConverter::postorder(const P4::IR::Declaration_Variable *decl) {
    ConversionTracer trace("Converting ", decl);

    auto type = getOrCreateType(decl);

    // TODO: Choose better insertion point for alloca (entry BB or so)
    auto var = builder.create<P4HIR::VariableOp>(getLoc(builder, decl), type,
                                                 builder.getStringAttr(decl->name.string_view()));

    if (const auto *init = decl->initializer) {
        var.setInit(true);
        builder.create<P4HIR::AssignOp>(getLoc(builder, init), getValue(decl->initializer), var);
    }

    setValue(decl, var);
}

void P4HIRConverter::postorder(const P4::IR::Cast *cast) {
    ConversionTracer trace("Converting ", cast);

    auto src = getValue(cast->expr);
    auto destType = getOrCreateType(cast->destType);

    setValue(cast, builder.create<P4HIR::CastOp>(getLoc(builder, cast), destType, src));
}

mlir::Value P4HIRConverter::emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind) {
    return builder.create<P4HIR::UnaryOp>(getLoc(builder, unop), kind, getValue(unop->expr));
}

mlir::Value P4HIRConverter::emitBinOp(const P4::IR::Operation_Binary *binop,
                                      P4HIR::BinOpKind kind) {
    return builder.create<P4HIR::BinOp>(getLoc(builder, binop), kind, getValue(binop->left),
                                        getValue(binop->right));
}

mlir::Value P4HIRConverter::emitConcatOp(const P4::IR::Concat *concatop) {
    return builder.create<P4HIR::ConcatOp>(getLoc(builder, concatop), getValue(concatop->left),
                                           getValue(concatop->right));
}

mlir::Value P4HIRConverter::emitCmp(const P4::IR::Operation_Relation *relop,
                                    P4HIR::CmpOpKind kind) {
    return builder.create<P4HIR::CmpOp>(getLoc(builder, relop), kind, getValue(relop->left),
                                        getValue(relop->right));
}

#define CONVERT_UNOP(Node, Kind)                                  \
    void P4HIRConverter::postorder(const P4::IR::Node *node) {    \
        ConversionTracer trace("Converting ", node);              \
        setValue(node, emitUnOp(node, P4HIR::UnaryOpKind::Kind)); \
    }

CONVERT_UNOP(Neg, Neg);
CONVERT_UNOP(UPlus, UPlus);
CONVERT_UNOP(Cmpl, Cmpl);
CONVERT_UNOP(LNot, LNot);

#undef CONVERT_UNOP

#define CONVERT_BINOP(Node, Kind)                                \
    void P4HIRConverter::postorder(const P4::IR::Node *node) {   \
        ConversionTracer trace("Converting ", node);             \
        setValue(node, emitBinOp(node, P4HIR::BinOpKind::Kind)); \
    }

CONVERT_BINOP(Mul, Mul);
CONVERT_BINOP(Div, Div);
CONVERT_BINOP(Mod, Mod);
CONVERT_BINOP(Add, Add);
CONVERT_BINOP(Sub, Sub);
CONVERT_BINOP(AddSat, AddSat);
CONVERT_BINOP(SubSat, SubSat);
CONVERT_BINOP(BOr, Or);
CONVERT_BINOP(BAnd, And);
CONVERT_BINOP(BXor, Xor);

#undef CONVERT_BINOP

void P4HIRConverter::postorder(const P4::IR::Concat *concat) {
    ConversionTracer trace("Converting ", concat);
    setValue(concat, emitConcatOp(concat));
}

#define CONVERT_CMP(Node, Kind)                                \
    void P4HIRConverter::postorder(const P4::IR::Node *node) { \
        ConversionTracer trace("Converting ", node);           \
        setValue(node, emitCmp(node, P4HIR::CmpOpKind::Kind)); \
    }

CONVERT_CMP(Equ, Eq);
CONVERT_CMP(Neq, Ne);
CONVERT_CMP(Lss, Lt);
CONVERT_CMP(Leq, Le);
CONVERT_CMP(Grt, Gt);
CONVERT_CMP(Geq, Ge);

#undef CONVERT_CMP

bool P4HIRConverter::preorder(const P4::IR::AssignmentStatement *assign) {
    ConversionTracer trace("Converting ", assign);

    // TODO: Handle slice on LHS here
    auto ref = resolveReference(assign->left);
    visit(assign->right);
    builder.create<P4HIR::AssignOp>(getLoc(builder, assign), getValue(assign->right), ref);
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LOr *lor) {
    ConversionTracer trace("Converting ", lor);

    // Lower a || b into a ? true : b
    visit(lor->left);

    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, lor), getValue(lor->left),
        [&](mlir::OpBuilder &b, mlir::Location loc) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, lor->left), getBoolConstant(loc, true));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(lor->right);
            b.create<P4HIR::YieldOp>(getEndLoc(builder, lor->right), getValue(lor->right));
        });

    setValue(lor, value.getResult());
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LAnd *land) {
    ConversionTracer trace("Converting ", land);

    // Lower a && b into a ? b : false
    visit(land->left);

    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, land), getValue(land->left),
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(land->right);
            b.create<P4HIR::YieldOp>(getEndLoc(builder, land->right), getValue(land->right));
        },
        [&](mlir::OpBuilder &b, mlir::Location loc) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, land->left), getBoolConstant(loc, false));
        });

    setValue(land, value.getResult());
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Mux *mux) {
    ConversionTracer trace("Converting ", mux);

    // Materialize condition first
    visit(mux->e0);

    // Make the value itself
    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, mux), getValue(mux->e0),
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(mux->e1);
            b.create<P4HIR::YieldOp>(getEndLoc(builder, mux->e1), getValue(mux->e1));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(mux->e2);
            b.create<P4HIR::YieldOp>(getEndLoc(builder, mux->e2), getValue(mux->e2));
        });

    setValue(mux, value.getResult());

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::IfStatement *ifs) {
    ConversionTracer trace("Converting ", ifs);

    // Materialize condition first
    visit(ifs->condition);

    // Create if itself
    builder.create<P4HIR::IfOp>(
        getLoc(builder, ifs), getValue(ifs->condition), ifs->ifFalse,
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(ifs->ifTrue);
            P4HIR::buildTerminatedBody(b, getEndLoc(builder, ifs->ifTrue));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            visit(ifs->ifFalse);
            P4HIR::buildTerminatedBody(b, getEndLoc(builder, ifs->ifFalse));
        });
    return false;
}

static llvm::SmallVector<mlir::DictionaryAttr, 4> convertParamDirections(
    const P4::IR::ParameterList *params, mlir::OpBuilder &b) {
    // Create attributes for directions
    llvm::SmallVector<mlir::DictionaryAttr, 4> argAttrs;
    for (const auto *p : params->parameters) {
        P4HIR::ParamDirection dir;
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

        argAttrs.emplace_back(b.getDictionaryAttr(
            b.getNamedAttr(P4HIR::FuncOp::getDirectionAttrName(),
                           P4HIR::ParamDirectionAttr::get(b.getContext(), dir))));
    }

    return argAttrs;
}

bool P4HIRConverter::preorder(const P4::IR::Function *f) {
    ConversionTracer trace("Converting ", f);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(f->type));
    const auto &params = f->getParameters()->parameters;

    auto argAttrs = convertParamDirections(f->getParameters(), builder);
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto func = builder.create<P4HIR::FuncOp>(getLoc(builder, f), f->name.string_view(), funcType,
                                              llvm::ArrayRef<mlir::NamedAttribute>(), argAttrs);
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
            builder.create<P4HIR::ReturnOp>(getEndLoc(builder, f));
        }
    }

    auto [it, inserted] = p4Symbols.try_emplace(f, mlir::SymbolRefAttr::get(func));
    BUG_CHECK(inserted, "duplicate translation of %1%", f);

    return false;
}

// We treat method as an external function (w/o body)
bool P4HIRConverter::preorder(const P4::IR::Method *m) {
    ConversionTracer trace("Converting ", m);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(m->type));

    auto argAttrs = convertParamDirections(m->getParameters(), builder);
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto func = builder.create<P4HIR::FuncOp>(getLoc(builder, m), m->name.string_view(), funcType,
                                              llvm::ArrayRef<mlir::NamedAttribute>(), argAttrs);

    auto [it, inserted] = p4Symbols.try_emplace(m, mlir::SymbolRefAttr::get(func));
    BUG_CHECK(inserted, "duplicate translation of %1%", m);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Action *act) {
    ConversionTracer trace("Converting ", act);

    // TODO: Actions might reference some control locals, we need to make
    // them visible somehow (e.g. via additional arguments)

    // FIXME: Get rid of typeMap: ensure action knows its type
    auto actType = mlir::cast<P4HIR::FuncType>(getOrCreateType(typeMap->getType(act, true)));
    const auto &params = act->getParameters()->parameters;

    auto argAttrs = convertParamDirections(act->getParameters(), builder);
    assert(actType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto action =
        P4HIR::FuncOp::buildAction(builder, getLoc(builder, act), act->name.string_view(), actType,
                                   llvm::ArrayRef<mlir::NamedAttribute>(), argAttrs);

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
        visit(act->body);

        // Check if body's last block is not terminated.
        mlir::Block &b = body.back();
        if (!b.mightHaveTerminator()) {
            builder.setInsertionPointToEnd(&b);
            builder.create<P4HIR::ReturnOp>(getEndLoc(builder, act));
        }
    }

    auto [it, inserted] = p4Symbols.try_emplace(act, mlir::SymbolRefAttr::get(action));
    BUG_CHECK(inserted, "duplicate translation of %1%", act);

    return false;
}

void P4HIRConverter::postorder(const P4::IR::ReturnStatement *ret) {
    ConversionTracer trace("Converting ", ret);

    // TODO: ReturnOp is a terminator, so it cannot be in the middle of block;
    // ensure nothing is created afterwards
    if (ret->expression) {
        auto retVal = getValue(ret->expression);
        builder.create<P4HIR::ReturnOp>(getLoc(builder, ret), retVal);
    } else {
        builder.create<P4HIR::ReturnOp>(getLoc(builder, ret));
    }
}

bool P4HIRConverter::preorder(const P4::IR::MethodCallExpression *mce) {
    ConversionTracer trace("Converting ", mce);
    const auto *instance =
        P4::MethodInstance::resolve(mce, this, typeMap, false, getChildContext());
    const auto &params = instance->originalMethodType->parameters->parameters;

    // TODO: Actions might have some parameters coming from control plane

    // Prepare call arguments. Note that this involves creating temporaries to
    // model copy-in/out semantics. To limit the lifetime of those temporaries, do
    // everything in the dedicated block scope. If there are no out parameters,
    // then emit everything direct.
    bool emitScope =
        std::any_of(params.begin(), params.end(), [](const auto *p) { return p->hasOut(); });
    auto convertCall = [&](mlir::OpBuilder &b, mlir::Type &resultType, mlir::Location loc) {
        llvm::SmallVector<mlir::Value, 4> operands;
        for (auto [idx, arg] : llvm::enumerate(*mce->arguments)) {
            ConversionTracer trace("Converting ", arg);
            mlir::Value argVal;
            switch (auto dir = params[idx]->direction) {
                case P4::IR::Direction::None:
                case P4::IR::Direction::In:
                    // Nothing to do special, just pass things direct
                    visit(arg->expression);
                    argVal = getValue(arg->expression);
                    break;
                case P4::IR::Direction::Out:
                case P4::IR::Direction::InOut: {
                    // Create temporary to hold the output value, initialize in case of inout
                    auto ref = resolveReference(arg->expression);
                    auto copyIn = b.create<P4HIR::VariableOp>(
                        loc, ref.getType(),
                        b.getStringAttr(
                            llvm::Twine(params[idx]->name.string_view()) +
                            (dir == P4::IR::Direction::InOut ? "_inout_arg" : "_out_arg")));

                    if (dir == P4::IR::Direction::InOut) {
                        copyIn.setInit(true);
                        b.create<P4HIR::AssignOp>(loc, b.create<P4HIR::ReadOp>(loc, ref), copyIn);
                    }
                    argVal = copyIn;
                    break;
                }
            }
            operands.push_back(argVal);
        }

        mlir::Value callResult;
        if (const auto *actCall = instance->to<P4::ActionCall>()) {
            auto actSym = p4Symbols.lookup(actCall->action);
            BUG_CHECK(actSym, "expected reference action to be converted: %1%", actCall->action);

            b.create<P4HIR::CallOp>(loc, actSym, operands);
        } else if (const auto *fCall = instance->to<P4::FunctionCall>()) {
            auto fSym = p4Symbols.lookup(fCall->function);
            auto callResultType = getOrCreateType(instance->originalMethodType->returnType);

            BUG_CHECK(fSym, "expected reference function to be converted: %1%", fCall->function);

            callResult = b.create<P4HIR::CallOp>(loc, fSym, callResultType, operands).getResult();
        } else if (const auto *fCall = instance->to<P4::ExternCall>()) {
            auto fSym = p4Symbols.lookup(fCall->method);
            auto callResultType = getOrCreateType(instance->originalMethodType->returnType);

            BUG_CHECK(fSym, "expected reference function to be converted: %1%", fCall->method);

            callResult = b.create<P4HIR::CallOp>(loc, fSym, callResultType, operands).getResult();
        } else {
            BUG("unsupported call type: %1%", mce);
        }

        for (auto [idx, arg] : llvm::enumerate(*mce->arguments)) {
            // Determine the direction of the parameter
            if (!params[idx]->hasOut()) continue;

            mlir::Value copyOut = operands[idx];
            mlir::Value dest = resolveReference(arg->expression);
            b.create<P4HIR::AssignOp>(
                getEndLoc(builder, mce),
                builder.create<P4HIR::ReadOp>(getEndLoc(builder, mce), copyOut), dest);
        }

        // If we are inside the scope, then build the yield of the call result
        if (emitScope) {
            if (callResult) {
                resultType = callResult.getType();
                b.create<P4HIR::YieldOp>(getEndLoc(b, mce), callResult);
            } else
                b.create<P4HIR::YieldOp>(getEndLoc(b, mce));
        } else {
            setValue(mce, callResult);
        }
    };

    if (emitScope) {
        auto scope = builder.create<P4HIR::ScopeOp>(getLoc(builder, mce), convertCall);
        setValue(mce, scope.getResults());
    } else {
        mlir::Type resultType;
        convertCall(builder, resultType, getLoc(builder, mce));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Member *m) {
    // This is just enum constant
    if (const auto *typeNameExpr = m->expr->to<P4::IR::TypeNameExpression>()) {
        auto type = getOrCreateType(typeNameExpr->typeName);
        BUG_CHECK((mlir::isa<P4HIR::EnumType, P4HIR::SerEnumType>(type)),
                  "unexpected type for expression %1%", typeNameExpr);

        setValue(m, builder.create<P4HIR::ConstOp>(
                        getLoc(builder, m),
                        P4HIR::EnumFieldAttr::get(type, m->member.name.string_view())));
        return false;
    }

    // Handle other members in postorder traversal
    return true;
}

void P4HIRConverter::postorder(const P4::IR::Member *m) {
    // Resolve member rvalue expression to something we can reason about
    // TODO: Likely we can do similar things for the majority of struct-like
    // types
    auto parentType = getOrCreateType(m->expr);
    if (auto structType = mlir::dyn_cast<P4HIR::StructType>(parentType)) {
        // We can access to parent using struct operations
        auto parent = getValue(m->expr);
        auto field = builder.create<P4HIR::StructExtractOp>(getLoc(builder, m), parent,
                                                            m->member.string_view());
        setValue(m, field.getResult());
    } else {
        BUG("cannot convert this member reference %1% (aka %2%) yet", m, dbp(m));
    }
}

bool P4HIRConverter::preorder(const P4::IR::StructExpression *str) {
    auto type = getOrCreateType(str->structType);
    llvm::SmallVector<mlir::Value, 4> fields;

    for (const auto *field : str->components) {
        visit(field->expression);
        fields.push_back(getValue(field->expression));
    }

    setValue(str, builder.create<P4HIR::StructOp>(getLoc(builder, str), type, fields).getResult());

    return false;
}

}  // namespace

namespace P4::P4MLIR {

mlir::OwningOpRef<mlir::ModuleOp> toMLIR(mlir::MLIRContext &context,
                                         const P4::IR::P4Program *program, P4::TypeMap *typeMap) {
    mlir::OpBuilder builder(&context);

    auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(moduleOp.getBody());

    if (auto sourceInfo = program->getSourceInfo(); sourceInfo.isValid()) {
        moduleOp.setSymName(sourceInfo.getSourceFile().string_view());
        moduleOp->setLoc(getLoc(builder, program));
    }
    P4HIRConverter conv(builder, typeMap);
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

}  // namespace P4::P4MLIR
