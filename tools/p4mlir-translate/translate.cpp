#include "translate.h"

#include <climits>

#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeMap.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/big_int.h"
#include "lib/indent.h"
#include "lib/log.h"
#include "llvm/Support/raw_ostream.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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
        LOG4(P4::IndentCtl::indent << Kind << dbp(node));
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

    mlir::Type getType() { return type; }
    bool setType(const P4::IR::Type *type, mlir::Type mlirType);

 private:
    P4HIRConverter &converter;
    mlir::Type type = nullptr;
};

class P4HIRConverter : public P4::Inspector, public P4::ResolutionContext {
    mlir::OpBuilder &builder;

    const P4::TypeMap *typeMap = nullptr;
    llvm::DenseMap<const P4::IR::Type *, mlir::Type> p4Types;
    // TODO: Implement unified constant map
    // using CTVOrExpr = std::variant<const P4::IR::CompileTimeValue *,
    //                                const P4::IR::Expression *>;
    // llvm::DenseMap<CTVOrExpr, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Expression *, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Node *, mlir::Value> p4Values;

    mlir::TypedAttr resolveConstant(const P4::IR::CompileTimeValue *ctv);
    mlir::TypedAttr resolveConstantExpr(const P4::IR::Expression *expr);
    mlir::Value resolveReference(const P4::IR::Node *node);

    mlir::Value getBoolConstant(mlir::Location loc, bool value) {
        auto boolType = P4HIR::BoolType::get(context());
        return builder.create<P4HIR::ConstOp>(loc, boolType,
                                              P4HIR::BoolAttr::get(context(), boolType, value));
    }

 public:
    P4HIRConverter(mlir::OpBuilder &builder, const P4::TypeMap *typeMap)
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
        return getOrCreateType(expr->type);
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

    mlir::TypedAttr getOrCreateConstantExpr(const P4::IR::Expression *expr) {
        auto cst = p4Constants.lookup(expr);
        if (cst) return cst;

        cst = resolveConstantExpr(expr);

        BUG_CHECK(cst, "expected %1% to be converted as constant", expr);
        return cst;
    }

    mlir::Value getValue(const P4::IR::Node *node) {
        // If this is a PathExpression, resolve it
        if (const auto *pe = node->to<P4::IR::PathExpression>()) {
            node = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
        }

        if (const auto *decl = node->to<P4::IR::Declaration_Variable>()) {
            // Getting value out of variable involves involves a load.
            auto alloca = p4Values.lookup(decl);
            BUG_CHECK(alloca, "expected %1% (aka %2%) to be converted", node, dbp(node));
            return builder.create<P4HIR::LoadOp>(getLoc(builder, node), alloca);
        }

        if (auto val = p4Values.lookup(node)) return val;

        BUG("expected %1% (aka %2%) to be converted", node, dbp(node));
    }

    mlir::Value setValue(const P4::IR::Node *node, mlir::Value value) {
        if (LOGGING(4)) {
            std::string s;
            llvm::raw_string_ostream os(s);
            value.print(os);
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
    bool preorder(const P4::IR::P4Action *a) override {
        // We cannot simply visit each node of the top-level block as
        // ResolutionContext would not be able to resolve declarations there
        // (sic!)
        visit(a->body);
        return false;
    }
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

    // Comparisons
    HANDLE_IN_POSTORDER(Equ)
    HANDLE_IN_POSTORDER(Neq)
    HANDLE_IN_POSTORDER(Leq)
    HANDLE_IN_POSTORDER(Lss)
    HANDLE_IN_POSTORDER(Grt)
    HANDLE_IN_POSTORDER(Geq)

    HANDLE_IN_POSTORDER(Cast)
    HANDLE_IN_POSTORDER(Declaration_Variable)

#undef HANDLE_IN_POSTORDER

    bool preorder(const P4::IR::Declaration_Constant *decl) override;
    bool preorder(const P4::IR::AssignmentStatement *assign) override;
    bool preorder(const P4::IR::LOr *lor) override;
    bool preorder(const P4::IR::LAnd *land) override;

    mlir::Value emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind);
    mlir::Value emitBinOp(const P4::IR::Operation_Binary *binop, P4HIR::BinOpKind kind);
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
    ConversionTracer trace("TypeConverting ", name);
    const auto *type = converter.resolveType(name);
    CHECK_NULL(type);
    visit(type);
    return false;
}

bool P4TypeConverter::setType(const P4::IR::Type *type, mlir::Type mlirType) {
    this->type = mlirType;
    converter.setType(type, mlirType);
    return false;
}

mlir::Value P4HIRConverter::resolveReference(const P4::IR::Node *node) {
    // If this is a PathExpression, resolve it
    if (const auto *pe = node->to<P4::IR::PathExpression>()) {
        node = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
    }

    // The result is expected to be an l-value
    auto ref = p4Values.lookup(node);
    BUG_CHECK(ref, "expected %1% (aka %2%) to be converted", node, dbp(node));
    BUG_CHECK(mlir::isa<P4HIR::ReferenceType>(ref.getType()),
              "expected reference type for node %1%", node);

    return ref;
}

mlir::TypedAttr P4HIRConverter::resolveConstant(const P4::IR::CompileTimeValue *ctv) {
    BUG("cannot resolve this constant yet %1%", ctv);
}

mlir::TypedAttr P4HIRConverter::resolveConstantExpr(const P4::IR::Expression *expr) {
    LOG4("Resolving " << dbp(expr) << " as constant expression");

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

    BUG("cannot resolve this constant expression yet %1%", expr);
}

mlir::Value P4HIRConverter::materializeConstantExpr(const P4::IR::Expression *expr) {
    ConversionTracer trace("Materializing constant expression ", expr);

    auto type = getOrCreateType(expr->type);
    auto init = getOrCreateConstantExpr(expr);
    auto loc = getLoc(builder, expr);

    // Hack: type inference sometimes keeps `Type_Unknown` for some constants, in such case
    // use type from the initializer
    if (mlir::isa<P4HIR::UnknownType>(type)) type = init.getType();

    auto val = builder.create<P4HIR::ConstOp>(loc, type, init);
    return setValue(expr, val);
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Constant *decl) {
    ConversionTracer trace("Converting ", decl);

    auto type = getOrCreateType(decl->type);
    auto init = getOrCreateConstantExpr(decl->initializer);
    auto loc = getLoc(builder, decl);

    auto val = builder.create<P4HIR::ConstOp>(loc, type, init);
    setValue(decl, val);

    return false;
}

void P4HIRConverter::postorder(const P4::IR::Declaration_Variable *decl) {
    ConversionTracer trace("Converting ", decl);

    const auto *init = decl->initializer;
    mlir::Type objectType;
    if (init) objectType = getOrCreateType(init);
    if (!objectType || mlir::isa<P4HIR::UnknownType>(objectType))
        objectType = getOrCreateType(decl->type);

    auto type = getOrCreateType(decl);

    // TODO: Choose better insertion point for alloca (entry BB or so)
    auto alloca = builder.create<P4HIR::AllocaOp>(getLoc(builder, decl), type, objectType,
                                                  decl->name.string_view());

    if (init) {
        alloca.setInit(true);
        builder.create<P4HIR::StoreOp>(getLoc(builder, init), getValue(decl->initializer), alloca);
    }

    setValue(decl, alloca);
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
    visit(assign->left);
    visit(assign->right);
    auto ref = resolveReference(assign->left);
    builder.create<P4HIR::StoreOp>(getLoc(builder, assign), getValue(assign->right), ref);
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LOr *lor) {
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

}  // namespace

namespace P4::P4MLIR {

mlir::OwningOpRef<mlir::ModuleOp> toMLIR(mlir::MLIRContext &context,
                                         const P4::IR::P4Program *program,
                                         const P4::TypeMap *typeMap) {
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
