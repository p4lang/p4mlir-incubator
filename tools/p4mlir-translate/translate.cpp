#include "translate.h"

#include <algorithm>
#include <climits>

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
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
    const auto &pos = sourceInfo.toPosition();

    return mlir::FileLineColLoc::get(builder.getStringAttr(pos.fileName.string_view()),
                                     pos.sourceLine, start.getColumnNumber());
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
class P4TypeConverter : public P4::Inspector, P4::ResolutionContext {
 public:
    explicit P4TypeConverter(P4HIRConverter &converter) : converter(converter) {}

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
    bool preorder(const P4::IR::Type_String *type) override;
    bool preorder(const P4::IR::Type_Unknown *type) override;
    bool preorder(const P4::IR::Type_Typedef *type) override;
    bool preorder(const P4::IR::Type_Name *name) override;
    bool preorder(const P4::IR::Type_Newtype *nt) override;
    bool preorder(const P4::IR::Type_Action *act) override;
    bool preorder(const P4::IR::Type_Void *v) override;
    bool preorder(const P4::IR::Type_Struct *s) override;
    bool preorder(const P4::IR::Type_Enum *e) override;
    bool preorder(const P4::IR::Type_Error *e) override;
    bool preorder(const P4::IR::Type_SerEnum *se) override;
    bool preorder(const P4::IR::Type_ActionEnum *e) override;
    bool preorder(const P4::IR::Type_Header *h) override;
    bool preorder(const P4::IR::Type_HeaderUnion *hu) override;
    bool preorder(const P4::IR::Type_BaseList *l) override;  // covers both Type_Tuple and Type_List
    bool preorder(const P4::IR::Type_Parser *p) override;
    bool preorder(const P4::IR::P4Parser *a) override;
    bool preorder(const P4::IR::Type_Control *c) override;
    bool preorder(const P4::IR::P4Control *c) override;
    bool preorder(const P4::IR::Type_Extern *e) override;
    bool preorder(const P4::IR::Type_Var *tv) override;
    bool preorder(const P4::IR::Type_Method *m) override;
    bool preorder(const P4::IR::Type_Specialized *t) override;
    bool preorder(const P4::IR::Type_SpecializedCanonical *t) override;
    bool preorder(const P4::IR::Type_Package *p) override;

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

    llvm::ScopedHashTable<const P4::IR::Node *, mlir::Value> p4Values;
    llvm::ScopedHashTable<const P4::IR::Node *, mlir::Value> controlPlaneValues;
    using ValueScope = llvm::ScopedHashTableScope<const P4::IR::Node *, mlir::Value>;

    using P4Symbol =
        std::variant<const P4::IR::P4Action *, const P4::IR::Function *, const P4::IR::Method *,
                     const P4::IR::P4Parser *, const P4::IR::P4Control *, const P4::IR::P4Table *>;
    // TODO: Implement better scoped symbol table
    llvm::DenseMap<P4Symbol, mlir::SymbolRefAttr> p4Symbols;

    mlir::TypedAttr resolveConstant(const P4::IR::CompileTimeValue *ctv);
    mlir::Value resolveReference(const P4::IR::Node *node, bool unchecked = false);

    mlir::Value getBoolConstant(mlir::Location loc, bool value) {
        return builder.create<P4HIR::ConstOp>(loc, P4HIR::BoolAttr::get(context(), value));
    }
    mlir::Value getStringConstant(mlir::Location loc, llvm::Twine &bytes) {
        return builder.create<P4HIR::ConstOp>(
            loc, mlir::StringAttr::get(bytes, P4HIR::StringType::get(context())));
    }

    mlir::TypedAttr getTypedConstant(mlir::Type type, mlir::Attribute constant) {
        if (mlir::isa<P4HIR::BoolType>(type)) return mlir::cast<P4HIR::BoolAttr>(constant);

        if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(type))
            return mlir::cast<P4HIR::IntAttr>(constant);

        if (mlir::isa<P4HIR::ErrorType>(type)) return mlir::cast<P4HIR::ErrorCodeAttr>(constant);

        return mlir::cast<P4HIR::AggAttr>(constant);
    }

    mlir::Value emitValidityConstant(mlir::Location loc, P4HIR::ValidityBit validityConstValue) {
        return builder.create<P4HIR::ConstOp>(
            loc, P4HIR::ValidityBitAttr::get(context(), validityConstValue));
    }

    void emitHeaderValidityBitAssignOp(mlir::Location loc, mlir::Value header,
                                       P4HIR::ValidityBit validityConstValue) {
        auto validityBitConstant = emitValidityConstant(loc, validityConstValue);
        auto validityBitRef =
            builder.create<P4HIR::StructExtractRefOp>(loc, header, P4HIR::HeaderType::validityBit);
        builder.create<P4HIR::AssignOp>(loc, validityBitConstant, validityBitRef);
    }

    P4HIR::CmpOp emitHeaderIsValidCmpOp(mlir::Location loc, mlir::Value header,
                                        P4HIR::ValidityBit compareWith) {
        mlir::Value validityBitValue;
        if (mlir::isa<P4HIR::ReferenceType>(header.getType())) {
            auto validityBitRef = builder.create<P4HIR::StructExtractRefOp>(
                loc, header, P4HIR::HeaderType::validityBit);
            validityBitValue = builder.create<P4HIR::ReadOp>(loc, validityBitRef);
        } else {
            validityBitValue =
                builder.create<P4HIR::StructExtractOp>(loc, header, P4HIR::HeaderType::validityBit);
        }
        auto validityConstant = emitValidityConstant(loc, compareWith);
        return builder.create<P4HIR::CmpOp>(loc, P4HIR::CmpOpKind::Eq, validityBitValue,
                                            validityConstant);
    }

    P4HIR::CmpOp emitHeaderUnionIsValidCmpOp(mlir::Location loc, mlir::Value headerUnion,
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
                header =
                    builder.create<P4HIR::StructExtractRefOp>(loc, headerUnion, fieldInfo.name);
            } else {
                header = builder.create<P4HIR::StructExtractOp>(loc, headerUnion, fieldInfo.name);
            }

            // Check if this member header is valid
            auto headerIsValid = emitHeaderIsValidCmpOp(loc, header, P4HIR::ValidityBit::Valid);

            // Create a ternary operation:
            // if this header is valid, return true,
            // otherwise check the next header in the header union
            auto ternaryOp = builder.create<P4HIR::TernaryOp>(
                loc, headerIsValid.getResult(),
                [&](mlir::OpBuilder &b, mlir::Location loc) {
                    // If this header is valid, return true
                    b.create<P4HIR::YieldOp>(loc, getBoolConstant(loc, true));
                },
                [&](mlir::OpBuilder &b, mlir::Location loc) {
                    // If this header is not valid, check the next header
                    b.create<P4HIR::YieldOp>(loc, buildNestedTernaryOp(fieldIndex + 1));
                });
            return ternaryOp.getResult();
        };

        // Start the recursive building from the first field
        auto isValid = buildNestedTernaryOp(0);

        // Return a comparison operation for consistency with other validity checks
        return builder.create<P4HIR::CmpOp>(
            loc, P4HIR::CmpOpKind::Eq, isValid,
            getBoolConstant(loc, compareWith == P4HIR::ValidityBit::Valid ? true : false));
    }

    void emitSetInvalidForAllHeaders(mlir::Location loc, mlir::Value headerUnion,
                                     const P4::cstring headerNameToSkip = nullptr) {
        auto headerUnionType = mlir::cast<P4HIR::HeaderUnionType>(getObjectType(headerUnion));
        llvm::for_each(headerUnionType.getFields(), [&](P4HIR::FieldInfo fieldInfo) {
            if (headerNameToSkip != fieldInfo.name.getValue()) {
                auto header =
                    builder.create<P4HIR::StructExtractRefOp>(loc, headerUnion, fieldInfo.name);
                emitHeaderValidityBitAssignOp(loc, header, P4HIR::ValidityBit::Invalid);
            }
        });
    }

 public:
    P4HIRConverter(mlir::OpBuilder &builder, P4::TypeMap *typeMap)
        : builder(builder), typeMap(typeMap) {
        CHECK_NULL(typeMap);
    }

    mlir::Type findType(const P4::IR::Type *type) const { return p4Types.lookup(type); }

    mlir::Type setType(const P4::IR::Type *type, mlir::Type mlirType) {
        auto [it, inserted] = p4Types.try_emplace(type, mlirType);
        BUG_CHECK(inserted, "duplicate conversion for %1%", type);

        return it->second;
    }

    mlir::Type getOrCreateConstructorType(const P4::IR::Type_Method *type) {
        // These things are a bit special: we keep names to simplify further
        // specialization during instantiation
        if (auto convertedType = findType(type)) return convertedType;

        ConversionTracer trace("Converting ctor type ", type);
        llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>, 4> argTypes;

        CHECK_NULL(type->parameters);

        mlir::Type resultType = getOrCreateType(type->returnType);

        for (const auto *p : type->parameters->parameters) {
            mlir::Type type = getOrCreateType(p->type);
            BUG_CHECK(p->direction == P4::IR::Direction::None, "expected directionless parameter");
            argTypes.emplace_back(builder.getStringAttr(p->name.string_view()), type);
        }

        auto mlirType = P4HIR::CtorType::get(argTypes, resultType);
        return setType(type, mlirType);
    }

    mlir::Type getOrCreateType(const P4::IR::Type *type) {
        P4TypeConverter cvt(*this);
        type->apply(cvt, getChildContext());
        return cvt.getType();
    }

    mlir::Type getOrCreateType(const P4::IR::Expression *expr) {
        return getOrCreateType(typeMap->getType(expr, true));
    }

    mlir::Type getOrCreateType(const P4::IR::Declaration_Variable *decl) {
        auto declType = getOrCreateType(decl->type);
        return P4HIR::ReferenceType::get(builder.getContext(), declType);
    }

    // Returns underlying type in case of something of serialized enum cate
    mlir::Type getIntType(const P4::IR::Type *type) {
        auto baseType = getOrCreateType(type);
        if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(baseType))
            baseType = aliasType.getAliasedType();
        if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(baseType))
            baseType = serEnumType.getType();
        return baseType;
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

    mlir::Value getValue(const P4::IR::Node *node, mlir::Type type = {}, bool unchecked = false) {
        // If this is a PathExpression, resolve it
        if (const auto *pe = node->to<P4::IR::PathExpression>()) {
            node = resolvePath(pe->path, false)->checkedTo<P4::IR::Declaration>();
        }

        auto val = p4Values.lookup(node);
        BUG_CHECK(val || unchecked, "expected %1% (aka %2%) to be converted", node, dbp(node));

        if (val && mlir::isa<P4HIR::ReferenceType>(val.getType()))
            // Getting value out of variable involves a load.
            val = builder.create<P4HIR::ReadOp>(getLoc(builder, node), val);

        if (type && val && val.getType() != type)
            val = builder.create<P4HIR::CastOp>(getLoc(builder, node), type, val);

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

        BUG_CHECK(!p4Values.count(node), "duplicate conversion of %1%");

        p4Values.insert(node, value);
        return value;
    }

    mlir::Value convert(const P4::IR::Node *node) {
        visit(node);
        return getValue(node);
    }

    mlir::Attribute convertAnnotationExpr(const P4::IR::Expression *ann);
    mlir::Attribute convert(const P4::IR::Annotation *anns);
    mlir::DictionaryAttr convert(const P4::IR::Vector<P4::IR::Annotation> &ann);
    llvm::SmallVector<mlir::DictionaryAttr, 4> convertParamAttributes(
        const P4::IR::ParameterList *params);

    mlir::MLIRContext *context() const { return builder.getContext(); }

    bool preorder(const P4::IR::Node *node) override {
        ::P4::error("P4 construct not yet supported: %1% (aka %2%)", node, dbp(node));
        return false;
    }

    bool preorder(const P4::IR::Type *type) override {
        ConversionTracer trace("Converting ", type);
        P4TypeConverter cvt(*this);
        type->apply(cvt, getChildContext());
        return false;
    }

    bool preorder(const P4::IR::P4Program *p) override {
        ValueScope scope(p4Values);

        // Explicitly visit child nodes to create top-level value scope
        visit(p->objects);

        return false;
    }
    bool preorder(const P4::IR::P4Action *a) override;
    bool preorder(const P4::IR::Function *f) override;

    bool preorder(const P4::IR::P4Parser *a) override;
    bool preorder(const P4::IR::ParserState *s) override;
    bool preorder(const P4::IR::SelectExpression *s) override;

    bool preorder(const P4::IR::Type_Extern *e) override;

    bool preorder(const P4::IR::P4Control *c) override;
    bool preorder(const P4::IR::P4Table *t) override;
    bool preorder(const P4::IR::Property *p) override;

    bool preorder(const P4::IR::Type_Package *e) override;

    bool preorder(const P4::IR::Method *m) override;
    bool preorder(const P4::IR::BlockStatement *block) override {
        ValueScope scope(p4Values);

        // If this is a top-level block where scope is implied (e.g. function,
        // action, certain statements) do not create explicit scope.
        if (getParent<P4::IR::BlockStatement>()) {
            auto annotations = convert(block->annotations);
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto scope = builder.create<P4HIR::ScopeOp>(
                getLoc(builder, block), annotations,
                [&](mlir::OpBuilder &, mlir::Location) {  // nothing is being yielded
                    visit(block->components);
                });
            builder.setInsertionPointToEnd(&scope.getScopeRegion().back());
            builder.create<P4HIR::YieldOp>(getEndLoc(builder, block));
        } else
            visit(block->components);
        return false;
    }
    bool preorder(const P4::IR::SwitchStatement *sw) override;

    bool preorder(const P4::IR::Constant *c) override {
        materializeConstantExpr(c);
        // FIXME: Serialized enum lowering might create references to the same
        // Constant (serenum member) from multiple scope. Allow multiple
        // materializations of the same constant until type inference will be
        // fixed.
        visitAgain();
        return false;
    }
    bool preorder(const P4::IR::BoolLiteral *b) override {
        materializeConstantExpr(b);
        visitAgain();
        return false;
    }
    bool preorder(const P4::IR::StringLiteral *s) override {
        materializeConstantExpr(s);
        visitAgain();
        return false;
    }
    bool preorder(const P4::IR::Cast *c) override {
        // Casts of constants could be used multiple times again and again. We need to visit
        // again in order to get them scoped properly
        if (c->expr->is<P4::IR::Literal>()) visitAgain();
        return true;
    }
    void postorder(const P4::IR::Cast *c) override;

    bool preorder(const P4::IR::PathExpression *e) override {
        // Should be resolved eslewhere
        return false;
    }
    bool preorder(const P4::IR::InvalidHeader *h) override {
        // Should be resolved eslewhere
        return false;
    }
    bool preorder(const P4::IR::InvalidHeaderUnion *hu) override {
        // Should be resolved eslewhere
        return false;
    }
    bool preorder(const P4::IR::Declaration_MatchKind *mk) override {
        // Should be resolved eslewhere
        return false;
    }
    bool preorder(const P4::IR::EmptyStatement *e) override {
        // Well, it's empty
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

    // Shift
    HANDLE_IN_POSTORDER(Shl)
    HANDLE_IN_POSTORDER(Shr)

    // Comparisons
    // == and != are a bit special and requires some postorder handling
    HANDLE_IN_POSTORDER(Leq)
    HANDLE_IN_POSTORDER(Lss)
    HANDLE_IN_POSTORDER(Grt)
    HANDLE_IN_POSTORDER(Geq)

    HANDLE_IN_POSTORDER(ReturnStatement)
    HANDLE_IN_POSTORDER(ExitStatement)
    HANDLE_IN_POSTORDER(ArrayIndex)
    HANDLE_IN_POSTORDER(Range)
    HANDLE_IN_POSTORDER(Mask)

#undef HANDLE_IN_POSTORDER

    void postorder(const P4::IR::Member *m) override;

    bool preorder(const P4::IR::Declaration_Constant *decl) override;
    bool preorder(const P4::IR::Declaration_Instance *decl) override;
    bool preorder(const P4::IR::Declaration_Variable *decl) override;
    bool preorder(const P4::IR::AssignmentStatement *assign) override;
    bool preorder(const P4::IR::Mux *mux) override;
    bool preorder(const P4::IR::Slice *slice) override;
    bool preorder(const P4::IR::LOr *lor) override;
    bool preorder(const P4::IR::LAnd *land) override;
    bool preorder(const P4::IR::IfStatement *ifs) override;
    bool preorder(const P4::IR::MethodCallStatement *) override {
        // We handle MethodCallExpression instead
        return true;
    }

    bool preorder(const P4::IR::MethodCallExpression *mce) override;
    bool preorder(const P4::IR::ConstructorCallExpression *cce) override;
    bool preorder(const P4::IR::StructExpression *str) override;
    bool preorder(const P4::IR::ListExpression *lst) override;
    bool preorder(const P4::IR::Member *m) override;
    bool preorder(const P4::IR::Equ *) override;
    bool preorder(const P4::IR::Neq *) override;
    void postorder(const P4::IR::Equ *) override;
    void postorder(const P4::IR::Neq *) override;

    mlir::Value emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind);
    mlir::Value emitBinOp(const P4::IR::Operation_Binary *binop, P4HIR::BinOpKind kind);
    mlir::Value emitConcatOp(const P4::IR::Concat *concatop);
    mlir::Value emitCmp(const P4::IR::Operation_Relation *relop, P4HIR::CmpOpKind kind);

 private:
    mlir::Value emitInvalidHeaderCmpOp(const P4::IR::Operation_Relation *p4RelationOp);
    mlir::Value emitInvalidHeaderUnionCmpOp(const P4::IR::Operation_Relation *p4RelationOp);
    mlir::Value emitHeaderBuiltInMethod(mlir::Location loc, const P4::BuiltInMethod *builtInMethod);
    mlir::Value emitHeaderUnionBuiltInMethod(mlir::Location loc,
                                             const P4::BuiltInMethod *builtInMethod);
    mlir::Type getObjectType(mlir::Value &value) {
        if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(value.getType()))
            return refType.getObjectType();
        return value.getType();
    }
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

bool P4TypeConverter::preorder(const P4::IR::Type_String *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::StringType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Unknown *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::UnknownType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Var *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    auto mlirType = P4HIR::TypeVarType::get(converter.context(), type->getVarName().string_view());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Typedef *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    mlir::Type mlirType = convert(type->type);

    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Name *name) {
    if ((this->type = converter.findType(name))) return false;

    ConversionTracer trace("Resolving type by name ", name);
    const auto *type = resolveType(name);
    CHECK_NULL(type);
    LOG4("Resolved to: " << dbp(type));

    mlir::Type mlirType = convert(type);

    return setType(name, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Newtype *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    mlir::Type aliasee = convert(type->type);

    auto annotations = converter.convert(type->annotations);
    auto mlirType =
        P4HIR::AliasType::get(converter.context(), type->name.string_view(), aliasee, annotations);

    return setType(type, mlirType);
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

    mlir::Type resultType =
        type->returnType ? convert(type->returnType) : P4HIR::VoidType::get(converter.context());

    for (const auto *p : type->parameters->parameters) {
        mlir::Type type = convert(p->type);
        argTypes.push_back(p->hasOut() ? P4HIR::ReferenceType::get(type) : type);
    }

    llvm::SmallVector<mlir::Type, 1> typeParameters;
    for (const auto *typeParam : type->getTypeParameters()->parameters)
        typeParameters.push_back(convert(typeParam));

    auto mlirType = P4HIR::FuncType::get(argTypes, resultType, typeParameters);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::P4Parser *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    return setType(type, convert(type->type));
}

bool P4TypeConverter::preorder(const P4::IR::Type_Parser *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (const auto *p : type->getApplyParameters()->parameters) {
        mlir::Type type = convert(p->type);
        argTypes.push_back(p->hasOut() ? P4HIR::ReferenceType::get(type) : type);
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType = P4HIR::ParserType::get(converter.context(), type->name.string_view(), argTypes,
                                           annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::P4Control *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    return setType(type, convert(type->type));
}

bool P4TypeConverter::preorder(const P4::IR::Type_Control *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (const auto *p : type->getApplyParameters()->parameters) {
        mlir::Type type = convert(p->type);
        argTypes.push_back(p->hasOut() ? P4HIR::ReferenceType::get(type) : type);
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType = P4HIR::ControlType::get(converter.context(), type->name.string_view(), argTypes,
                                            annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Package *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    llvm::SmallVector<mlir::Type, 1> typeParameters;
    for (const auto *typeParam : type->typeParameters->parameters)
        typeParameters.push_back(convert(typeParam));

    auto annotations = converter.convert(type->annotations);
    auto mlirType = P4HIR::PackageType::get(converter.context(), type->name.string_view(),
                                            typeParameters, annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Extern *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    BUG_CHECK(type->typeParameters->empty(), "expected no type parameters for ext");

    auto annotations = converter.convert(type->annotations);
    auto mlirType =
        P4HIR::ExternType::get(converter.context(), type->name.string_view(), annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Specialized *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    llvm::SmallVector<mlir::Type, 1> typeArguments;
    for (const auto *typeArg : *type->arguments) typeArguments.push_back(convert(typeArg));

    mlir::Type mlirType;
    const auto *baseType = resolveType(type->baseType);
    if (const auto *extType = baseType->to<P4::IR::Type_Extern>()) {
        auto annotations = converter.convert(extType->annotations);
        mlirType = P4HIR::ExternType::get(converter.context(), extType->name.string_view(),
                                          typeArguments, annotations);
    } else if (const auto *pkgType = baseType->to<P4::IR::Type_Package>()) {
        auto annotations = converter.convert(pkgType->annotations);
        mlirType = P4HIR::PackageType::get(converter.context(), pkgType->name.string_view(),
                                           typeArguments, annotations);
    } else if (baseType->is<P4::IR::Type_Parser>() || baseType->is<P4::IR::Type_Control>()) {
        // Parser and control type might be generic in package block and ctor arguments
        mlir::Type baseMlirType = convert(baseType);
        auto annotations =
            converter.convert(baseType->checkedTo<P4::IR::Type_ArchBlock>()->annotations);
        if (auto parserType = llvm::dyn_cast<P4HIR::ParserType>(baseMlirType)) {
            mlirType = P4HIR::ParserType::get(converter.context(), parserType.getName(),
                                              parserType.getInputs(), typeArguments, annotations);
        } else {
            auto controlType = llvm::dyn_cast<P4HIR::ControlType>(baseMlirType);
            mlirType = P4HIR::ControlType::get(converter.context(), controlType.getName(),
                                               controlType.getInputs(), typeArguments, annotations);
        }
    } else
        BUG("Expected extern or package specialization: %1%", baseType);

    return setType(type, mlirType);
}

// TODO: This should never exist outside type inference stage...
bool P4TypeConverter::preorder(const P4::IR::Type_SpecializedCanonical *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    llvm::SmallVector<mlir::Type, 1> typeArguments;
    for (const auto *typeArg : *type->arguments) typeArguments.push_back(convert(typeArg));

    mlir::Type mlirType;
    const auto *baseType = resolveType(type->baseType);
    if (const auto *extType = baseType->to<P4::IR::Type_Extern>()) {
        auto annotations = converter.convert(extType->annotations);
        mlirType = P4HIR::ExternType::get(converter.context(), extType->name.string_view(),
                                          typeArguments, annotations);
    } else if (const auto *pkgType = baseType->to<P4::IR::Type_Package>()) {
        auto annotations = converter.convert(pkgType->annotations);
        mlirType = P4HIR::PackageType::get(converter.context(), pkgType->name.string_view(),
                                           typeArguments, annotations);
    } else if (baseType->is<P4::IR::Type_Parser>() || baseType->is<P4::IR::Type_Control>()) {
        // Parser and control type might be generic in package block and ctor arguments
        mlir::Type baseMlirType = convert(baseType);
        auto annotations =
            converter.convert(baseType->checkedTo<P4::IR::Type_ArchBlock>()->annotations);
        if (auto parserType = llvm::dyn_cast<P4HIR::ParserType>(baseMlirType)) {
            mlirType = P4HIR::ParserType::get(converter.context(), parserType.getName(),
                                              parserType.getInputs(), typeArguments, annotations);
        } else {
            auto controlType = llvm::dyn_cast<P4HIR::ControlType>(baseMlirType);
            mlirType = P4HIR::ControlType::get(converter.context(), controlType.getName(),
                                               controlType.getInputs(), typeArguments, annotations);
        }
    } else
        BUG("Expected extern or package specialization: %1%", baseType);

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
    llvm::SmallVector<P4HIR::FieldInfo, 4> fields;
    for (const auto *field : type->fields) {
        auto fieldAnnotations = converter.convert(field->annotations);
        fields.emplace_back(mlir::StringAttr::get(converter.context(), field->name.string_view()),
                            convert(field->type), fieldAnnotations);
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType =
        P4HIR::StructType::get(converter.context(), type->name.string_view(), fields, annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Header *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<P4HIR::FieldInfo, 4> fields;
    for (const auto *field : type->fields) {
        auto fieldAnnotations = converter.convert(field->annotations);
        fields.emplace_back(mlir::StringAttr::get(converter.context(), field->name.string_view()),
                            convert(field->type), fieldAnnotations);
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType =
        P4HIR::HeaderType::get(converter.context(), type->name.string_view(), fields, annotations);

    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_HeaderUnion *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<P4HIR::FieldInfo, 4> fields;
    for (const auto *field : type->fields) {
        auto fieldAnnotations = converter.convert(field->annotations);

        fields.emplace_back(mlir::StringAttr::get(converter.context(), field->name.string_view()),
                            convert(field->type), fieldAnnotations);
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType = P4HIR::HeaderUnionType::get(converter.context(), type->name.string_view(),
                                                fields, annotations);

    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Enum *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Attribute, 4> cases;
    for (const auto *field : type->members) {
        cases.push_back(mlir::StringAttr::get(converter.context(), field->name.string_view()));
    }

    auto annotations = converter.convert(type->annotations);
    auto mlirType =
        P4HIR::EnumType::get(converter.context(), type->name.string_view(), cases, annotations);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_ActionEnum *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Attribute, 4> cases;
    for (const auto *action : type->actionList->actionList) {
        cases.push_back(
            mlir::StringAttr::get(converter.context(), action->getName().string_view()));
    }
    auto mlirType = P4HIR::EnumType::get(converter.context(), "action_enum", cases);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Error *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Attribute, 4> cases;
    for (const auto *field : type->members) {
        cases.push_back(mlir::StringAttr::get(converter.context(), field->name.string_view()));
    }
    auto mlirType = P4HIR::ErrorType::get(converter.context(),
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

    auto annotations = converter.convert(type->annotations);
    auto mlirType = P4HIR::SerEnumType::get(type->name.string_view(), enumType, cases, annotations);

    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_BaseList *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<mlir::Type, 4> fields;
    for (const auto *field : type->components) {
        fields.push_back(convert(field));
    }

    auto mlirType = mlir::TupleType::get(converter.context(), fields);
    return setType(type, mlirType);
}

bool P4TypeConverter::setType(const P4::IR::Type *type, mlir::Type mlirType) {
    BUG_CHECK(mlirType, "empty type conversion for %1% (aka %2%)", type, dbp(type));
    this->type = mlirType;
    LOG4("type set for: " << dbp(type));
    converter.setType(type, mlirType);
    return false;
}

mlir::Type P4TypeConverter::convert(const P4::IR::Type *type) {
    if ((this->type = converter.findType(type))) return getType();

    visit(type);
    return getType();
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
    auto ref = p4Values.lookup(node);
    if (ref) return ref;

    ConversionTracer trace("Resolving reference ", node);
    // Check if this is a reference to a member of something we can recognize
    if (const auto *m = node->to<P4::IR::Member>()) {
        auto base = resolveReference(m->expr, unchecked);
        mlir::Value fieldRef;
        if (mlir::isa<P4HIR::ReferenceType>(base.getType()))
            fieldRef = builder
                           .create<P4HIR::StructExtractRefOp>(getLoc(builder, m), base,
                                                              m->member.string_view())
                           .getResult();
        else
            fieldRef = builder
                           .create<P4HIR::StructExtractOp>(getLoc(builder, m), base,
                                                           m->member.string_view())
                           .getResult();

        return setValue(m, fieldRef);
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
    if (const auto *s = expr->to<P4::IR::StringLiteral>()) {
        auto type = P4HIR::StringType::get(context());

        return setConstantExpr(s, mlir::StringAttr::get(s->value.string_view(), type));
    }
    if (const auto *cast = expr->to<P4::IR::Cast>()) {
        mlir::Type destType = getOrCreateType(cast);
        mlir::Type srcType = getOrCreateType(cast->expr);
        // Fold equal-type casts (e.g. due to typedefs)
        if (destType == srcType) return setConstantExpr(expr, getOrCreateConstantExpr(cast->expr));

        // Fold some conversions
        if (auto destBitsType = mlir::dyn_cast<P4HIR::BitsType>(destType)) {
            if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(srcType)) {
                auto castee = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(cast->expr));
                return setConstantExpr(
                    expr,
                    P4HIR::IntAttr::get(context(), destBitsType,
                                        castee.getValue().zextOrTrunc(destBitsType.getWidth())));
            }
            if (mlir::isa<P4HIR::SerEnumType>(srcType)) {
                auto castee = mlir::cast<P4HIR::EnumFieldAttr>(getOrCreateConstantExpr(cast->expr));
                auto enumType = mlir::cast<P4HIR::SerEnumType>(castee.getType());
                auto casteeVal =
                    mlir::cast<P4HIR::IntAttr>(enumType.valueOf(castee.getField().getValue()));
                return setConstantExpr(
                    expr,
                    P4HIR::IntAttr::get(context(), destBitsType,
                                        casteeVal.getValue().zextOrTrunc(destBitsType.getWidth())));
            }
        }

        // Handle casts to aliased types
        if (auto destAliasType = mlir::dyn_cast<P4HIR::AliasType>(destType)) {
            assert(destAliasType.getAliasedType() == srcType && "expected aliased types match");
            if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(srcType)) {
                auto castee = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(cast->expr));
                return setConstantExpr(
                    expr, P4HIR::IntAttr::get(context(), destAliasType, castee.getValue()));
            }
            if (auto srcBoolType = mlir::dyn_cast<P4HIR::BoolType>(srcType)) {
                auto castee = mlir::cast<P4HIR::BoolAttr>(getOrCreateConstantExpr(cast->expr));
                return setConstantExpr(
                    expr, P4HIR::BoolAttr::get(context(), destAliasType, castee.getValue()));
            }
        }
    }
    if (const auto *lst = expr->to<P4::IR::ListExpression>()) {
        auto type = getOrCreateType(lst->type);
        llvm::SmallVector<mlir::Attribute, 4> fields;
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
    auto loc = getLoc(builder, expr);

    auto val = builder.create<P4HIR::ConstOp>(loc, init);
    return setValue(expr, val);
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Constant *decl) {
    ConversionTracer trace("Converting ", decl);

    auto annotations = convert(decl->annotations);

    auto init = getOrCreateConstantExpr(decl->initializer);
    auto loc = getLoc(builder, decl);

    auto val = builder.create<P4HIR::ConstOp>(loc, init, decl->name.string_view(), annotations);
    setValue(decl, val);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Variable *decl) {
    ConversionTracer trace("Converting ", decl);

    auto annotations = convert(decl->annotations);

    auto type = getOrCreateType(decl);

    visit(decl->initializer);

    // TODO: Choose better insertion point for alloca (entry BB or so)
    auto var = builder.create<P4HIR::VariableOp>(getLoc(builder, decl), type,
                                                 decl->name.string_view(), annotations);

    if (const auto *init = decl->initializer) {
        var.setInit(true);
        auto loc = getLoc(builder, init);
        if (init->is<P4::IR::InvalidHeader>()) {
            // Handle special case of InvalidHeader initializer, we'd want to
            // initialize validity bit only, not the whole header
            emitHeaderValidityBitAssignOp(loc, var, P4HIR::ValidityBit::Invalid);
        } else if (init->is<P4::IR::InvalidHeaderUnion>()) {
            emitSetInvalidForAllHeaders(loc, var);
        } else {
            auto objType = llvm::cast<P4HIR::ReferenceType>(type).getObjectType();
            builder.create<P4HIR::AssignOp>(loc, getValue(decl->initializer, objType), var);
        }
    }

    setValue(decl, var);

    return false;
}

void P4HIRConverter::postorder(const P4::IR::Cast *cast) {
    ConversionTracer trace("Converting ", cast);

    auto src = getValue(cast->expr);
    auto destType = getOrCreateType(cast->destType);

    setValue(cast, builder.create<P4HIR::CastOp>(getLoc(builder, cast), destType, src));
}

bool P4HIRConverter::preorder(const P4::IR::Slice *slice) {
    ConversionTracer trace("Converting ", slice);

    auto maybeRef = resolveReference(slice->e0, /* unchecked */ true);
    auto destType = getOrCreateType(slice->type);

    mlir::Value sliceVal;
    if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(maybeRef.getType());
        refType && mlir::isa<P4HIR::BitsType>(refType.getObjectType())) {
        sliceVal = builder.create<P4HIR::SliceRefOp>(getLoc(builder, slice), destType, maybeRef,
                                                     slice->getH(), slice->getL());
    } else {
        sliceVal = builder.create<P4HIR::SliceOp>(getLoc(builder, slice), destType,
                                                  getValue(slice->e0, getIntType(slice->e0->type)),
                                                  slice->getH(), slice->getL());
    }

    setValue(slice, sliceVal);
    return false;
}

mlir::Value P4HIRConverter::emitUnOp(const P4::IR::Operation_Unary *unop, P4HIR::UnaryOpKind kind) {
    auto type = getOrCreateType(unop->type);
    return builder.create<P4HIR::UnaryOp>(getLoc(builder, unop), kind, getValue(unop->expr, type));
}

mlir::Value P4HIRConverter::emitBinOp(const P4::IR::Operation_Binary *binop,
                                      P4HIR::BinOpKind kind) {
    auto type = getOrCreateType(binop->type);
    return builder.create<P4HIR::BinOp>(getLoc(builder, binop), kind, getValue(binop->left, type),
                                        getValue(binop->right, type));
}

mlir::Value P4HIRConverter::emitConcatOp(const P4::IR::Concat *concatop) {
    return builder.create<P4HIR::ConcatOp>(getLoc(builder, concatop), getValue(concatop->left),
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

    return builder.create<P4HIR::CmpOp>(getLoc(builder, relop), kind, lhs, rhs);
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

#define CONVERT_SHL_SHR_OP(P4C_Shift, PHIR_Shift)                                         \
    void P4HIRConverter::postorder(const P4::IR::P4C_Shift *op) {                         \
        ConversionTracer trace("Converting ", op);                                        \
        auto type = getOrCreateType(op->type);                                            \
        auto intType = getIntType(op->right->type);                                       \
        auto result = builder.create<P4HIR::PHIR_Shift>(                                  \
            getLoc(builder, op), getValue(op->left, type), getValue(op->right, intType)); \
        setValue(op, result);                                                             \
    }

CONVERT_SHL_SHR_OP(Shl, ShlOp);
CONVERT_SHL_SHR_OP(Shr, ShrOp);

#undef CONVERT_SHL_SHR_OP

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

mlir::Value P4HIRConverter::emitInvalidHeaderCmpOp(const P4::IR::Operation_Relation *relOp) {
    auto loc = getLoc(builder, relOp);
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
    auto loc = getLoc(builder, relOp);
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

    auto loc = getLoc(builder, assign);

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
        builder.create<P4HIR::AssignSliceOp>(loc, convert(assign->right), ref, h, l);
    } else {
        auto ref = resolveReference(assign->left);
        auto objectType = mlir::cast<P4HIR::ReferenceType>(ref.getType()).getObjectType();
        visit(assign->right);
        builder.create<P4HIR::AssignOp>(loc, getValue(assign->right, objectType), ref);
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LOr *lor) {
    ConversionTracer trace("Converting ", lor);

    // Lower a || b into a ? true : b
    auto lhs = convert(lor->left);

    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, lor), lhs,
        [&](mlir::OpBuilder &b, mlir::Location loc) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, lor->left), getBoolConstant(loc, true));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, lor->right), convert(lor->right));
        });

    setValue(lor, value.getResult());
    return false;
}

bool P4HIRConverter::preorder(const P4::IR::LAnd *land) {
    ConversionTracer trace("Converting ", land);

    // Lower a && b into a ? b : false
    auto lhs = convert(land->left);

    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, land), lhs,
        [&](mlir::OpBuilder &b, mlir::Location) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, land->right), convert(land->right));
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
    auto cond = convert(mux->e0);

    // Make the value itself
    auto value = builder.create<P4HIR::TernaryOp>(
        getLoc(builder, mux), cond,
        [&](mlir::OpBuilder &b, mlir::Location) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, mux->e1), convert(mux->e1));
        },
        [&](mlir::OpBuilder &b, mlir::Location) {
            b.create<P4HIR::YieldOp>(getEndLoc(builder, mux->e2), convert(mux->e2));
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
    builder.create<P4HIR::IfOp>(
        getLoc(builder, ifs), cond, ifs->ifFalse,
        [&](mlir::OpBuilder &b, mlir::Location) {
            ValueScope scope(p4Values);

            visit(ifs->ifTrue);
            P4HIR::buildTerminatedBody(b, getEndLoc(builder, ifs->ifTrue));
        },
        thenAnnotations,
        [&](mlir::OpBuilder &b, mlir::Location) {
            ValueScope scope(p4Values);

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
    ValueScope scope(p4Values);

    auto annotations = convert(f->annotations);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(f->type));
    const auto &params = f->getParameters()->parameters;

    auto argAttrs = convertParamAttributes(f->getParameters());
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto loc = getLoc(builder, f);

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

            ovl = builder.create<P4HIR::OverloadSetOp>(loc, origSymName);
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

    auto func = builder.create<P4HIR::FuncOp>(loc, symName, funcType,
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
            builder.create<P4HIR::ImplicitReturnOp>(getEndLoc(builder, f));
        }
    }

    auto [it, inserted] = p4Symbols.try_emplace(f, mlir::SymbolRefAttr::get(origSymName));
    BUG_CHECK(inserted, "duplicate translation of %1%", f);

    return false;
}

// We treat method as an external function (w/o body)
bool P4HIRConverter::preorder(const P4::IR::Method *m) {
    // Special case: do not emit declaration for verify
    if (m->name == P4::IR::ParserState::verify) return false;

    ConversionTracer trace("Converting ", m);
    ValueScope scope(p4Values);

    auto annotations = convert(m->annotations);

    auto funcType = mlir::cast<P4HIR::FuncType>(getOrCreateType(m->type));

    auto argAttrs = convertParamAttributes(m->getParameters());
    assert(funcType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto loc = getLoc(builder, m);

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

            ovl = builder.create<P4HIR::OverloadSetOp>(loc, origSymName);
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

    builder.create<P4HIR::FuncOp>(loc, symName, funcType,
                                  /* isExternal */ true, argAttrs, annotations);

    auto [it, inserted] = p4Symbols.try_emplace(m, mlir::SymbolRefAttr::get(origSymName));
    BUG_CHECK(inserted, "duplicate translation of %1%", m);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Action *act) {
    ConversionTracer trace("Converting ", act);
    ValueScope scope(p4Values);

    // FIXME: Get rid of typeMap: ensure action knows its type
    auto actType = mlir::cast<P4HIR::FuncType>(getOrCreateType(typeMap->getType(act, true)));
    const auto &params = act->getParameters()->parameters;

    auto annotations = convert(act->annotations);

    auto argAttrs = convertParamAttributes(act->getParameters());
    assert(actType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto action = P4HIR::FuncOp::buildAction(builder, getLoc(builder, act), act->name.string_view(),
                                             actType, argAttrs, annotations);

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
            builder.create<P4HIR::ImplicitReturnOp>(getEndLoc(builder, act));
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
void P4HIRConverter::postorder(const P4::IR::ExitStatement *ex) {
    ConversionTracer trace("Converting ", ex);

    builder.create<P4HIR::ExitOp>(getLoc(builder, ex));
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
            auto loc = getLoc(builder, mce);

            // Check if this is a method call on a header or header union
            if (const auto *member = mce->method->to<P4::IR::Member>()) {
                // Check if it's a reference to a header or header union
                if (member->expr->type->is<P4::IR::Type_HeaderUnion>()) {
                    setValue(mce, emitHeaderUnionBuiltInMethod(loc, bCall));
                } else if (member->expr->type->to<P4::IR::Type_Header>()) {
                    setValue(mce, emitHeaderBuiltInMethod(loc, bCall));
                }
            }
            return;
        }
        // Another special case: some builtin methods are actually externs
        if (const auto *eCall = instance->to<P4::ExternCall>()) {
            // Transform verify call inside parser into proper transition op
            if (eCall->method->name == P4::IR::ParserState::verify &&
                isInContext<P4::IR::ParserState>()) {
                LOG4("Resolving verify() call");

                // Lower condition. TBD: This is incorrect if arguments are
                // passed by name.
                const auto *condExpr = mce->arguments->at(0)->expression;

                // Emit conditional reject op. Parser control flow
                // simplification should take care of this and emit a select
                // transition to a reject-with-error state
                auto condValue = convert(condExpr);
                condValue = builder
                                .create<P4HIR::UnaryOp>(getLoc(builder, condExpr),
                                                        P4HIR::UnaryOpKind::LNot, condValue)
                                .getResult();
                builder.create<P4HIR::IfOp>(
                    getEndLoc(builder, eCall->method), condValue, false,
                    [&](mlir::OpBuilder &b, mlir::Location) {
                        const auto *err = mce->arguments->at(1)->expression;
                        auto errCode =
                            mlir::cast<P4HIR::ErrorCodeAttr>(getOrCreateConstantExpr(err));
                        b.create<P4HIR::ParserRejectOp>(getEndLoc(b, err), errCode);
                    });
                return;
            }
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
            switch (auto dir = param->direction) {
                case P4::IR::Direction::None:
                case P4::IR::Direction::In: {
                    auto paramType = getOrCreateType(param->type);

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
                        auto copyIn = b.create<P4HIR::VariableOp>(
                            loc, P4HIR::ReferenceType::get(sliceType),
                            llvm::Twine(param->name.string_view()) +
                                (dir == P4::IR::Direction::InOut ? "_inout_arg" : "_out_arg"));

                        if (dir == P4::IR::Direction::InOut) {
                            copyIn.setInit(true);
                            b.create<P4HIR::AssignOp>(
                                loc,
                                b.create<P4HIR::SliceRefOp>(loc, sliceType, ref, slice->getH(),
                                                            slice->getL()),
                                copyIn);
                        }
                        argVal = copyIn;
                    } else {
                        auto ref = resolveReference(arg->expression);
                        auto copyIn = b.create<P4HIR::VariableOp>(
                            loc, ref.getType(),
                            llvm::Twine(param->name.string_view()) +
                                (dir == P4::IR::Direction::InOut ? "_inout_arg" : "_out_arg"));

                        if (dir == P4::IR::Direction::InOut) {
                            copyIn.setInit(true);
                            b.create<P4HIR::AssignOp>(loc, b.create<P4HIR::ReadOp>(loc, ref),
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
                // where argument might come from control plane. Grab
                // placeholder for it.
                // TBD: Handle @optional
                BUG_CHECK(param->direction == P4::IR::Direction::None,
                          "control plane values should be directionless");

                auto placeholder = controlPlaneValues.lookup(param);
                BUG_CHECK(placeholder, "control plane value must be populate");
                operands.push_back(placeholder);
            }
        }

        if (const auto *actCall = instance->to<P4::ActionCall>()) {
            LOG4("resolved as action call");
            auto actSym = p4Symbols.lookup(actCall->action);
            BUG_CHECK(actSym, "expected reference action to be converted: %1%", actCall->action);

            BUG_CHECK(mce->typeArguments->empty(), "expected action to be specialized");

            b.create<P4HIR::CallOp>(loc, actSym, operands);
        } else if (const auto *fCall = instance->to<P4::FunctionCall>()) {
            LOG4("resolved as function call");
            auto fSym = p4Symbols.lookup(fCall->function);
            auto callResultType = getOrCreateType(instance->actualMethodType->returnType);

            BUG_CHECK(fSym, "expected reference function to be converted: %1%", fCall->function);
            BUG_CHECK(mce->typeArguments->empty(), "expected function to be specialized");

            callResult = b.create<P4HIR::CallOp>(loc, fSym, callResultType, operands).getResult();
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

            callResult = b.create<P4HIR::CallOp>(loc, fSym, callResultType, typeArguments, operands)
                             .getResult();
        } else if (const auto *aCall = instance->to<P4::ApplyMethod>()) {
            LOG4("resolved as apply");
            BUG_CHECK(mce->typeArguments->empty(), "expected decl to be specialized");
            // Apply of something instantiated
            if (auto *decl = aCall->object->to<P4::IR::Declaration_Instance>()) {
                auto val = getValue(decl);
                b.create<P4HIR::ApplyOp>(loc, val, operands);
            } else if (auto *table = aCall->object->to<P4::IR::P4Table>()) {
                auto tSym = p4Symbols.lookup(table);
                auto applyResultType = getOrCreateType(instance->actualMethodType->returnType);
                callResult = b.create<P4HIR::TableApplyOp>(loc, applyResultType, tSym).getResult();
            } else
                BUG("Unsuported apply: %1% (aka %2%)", aCall->object, dbp(aCall->object));
        } else if (const auto *fCall = instance->to<P4::ExternMethod>()) {
            LOG4("resolved as extern method call ");

            // We need to do some weird dance to resolve method call, as fCall->method will not
            // resolve to a known symbol.
            const auto *member = mce->method->checkedTo<P4::IR::Member>();
            auto callResultType = getOrCreateType(instance->actualMethodType->returnType);
            auto externName = builder.getStringAttr(fCall->actualExternType->name.string_view());
            auto methodName =
                mlir::SymbolRefAttr::get(builder.getContext(), member->member.string_view());
            auto fullMethodName =
                mlir::SymbolRefAttr::get(builder.getContext(), externName, {methodName});

            // TODO: Move to common method
            llvm::SmallVector<mlir::Type> typeArguments;
            for (const auto *type : *mce->typeArguments) {
                typeArguments.push_back(getOrCreateType(type));
            }

            callResult = b.create<P4HIR::CallMethodOp>(loc, callResultType, getValue(member->expr),
                                                       fullMethodName, typeArguments, operands)
                             .getResult();
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
                b.create<P4HIR::AssignSliceOp>(
                    getEndLoc(builder, mce),
                    builder.create<P4HIR::ReadOp>(getEndLoc(builder, mce), copyOut), dest,
                    slice->getH(), slice->getL());
            } else {
                mlir::Value dest = resolveReference(arg->expression);
                b.create<P4HIR::AssignOp>(
                    getEndLoc(builder, mce),
                    builder.create<P4HIR::ReadOp>(getEndLoc(builder, mce), copyOut), dest);
            }
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
    if (const auto *spec = type->to<P4::IR::Type_Specialized>()) {
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
            // TODO: Handle @optional
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

        auto instance = builder.create<P4HIR::InstantiateOp>(
            getLoc(builder, cce), resultType, parserSym.getRootReference(), operands,
            parserSym.getRootReference(), mlir::DictionaryAttr());
        setValue(cce, instance.getResult());
    } else if (const auto *control = type->to<P4::IR::P4Control>()) {
        LOG4("resolved as control instantiation");
        auto controlSym = p4Symbols.lookup(control);
        BUG_CHECK(controlSym, "expected reference control to be converted: %1%", dbp(control));

        auto instance = builder.create<P4HIR::InstantiateOp>(
            getLoc(builder, cce), resultType, controlSym.getRootReference(), operands,
            controlSym.getRootReference(), mlir::DictionaryAttr());
        setValue(cce, instance.getResult());
    } else if (const auto *ext = type->to<P4::IR::Type_Extern>()) {
        LOG4("resolved as extern instantiation");

        auto externName = builder.getStringAttr(ext->name.string_view());
        auto instance =
            builder.create<P4HIR::InstantiateOp>(getLoc(builder, cce), resultType, externName,
                                                 operands, externName, mlir::DictionaryAttr());
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
        auto loc = getLoc(builder, m);

        if (mlir::isa<P4HIR::ErrorType>(type))
            setValue(m, builder.create<P4HIR::ConstOp>(
                            loc, P4HIR::ErrorCodeAttr::get(type, m->member.name.string_view())));
        else if (mlir::isa<P4HIR::EnumType>(type))
            setValue(m, builder.create<P4HIR::ConstOp>(
                            loc, P4HIR::EnumFieldAttr::get(type, m->member.name.string_view())));
        else if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(type)) {
            setValue(m, builder.create<P4HIR::ConstOp>(
                            loc, P4HIR::EnumFieldAttr::get(type, m->member.name.string_view())));
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
    if (mlir::isa<P4HIR::StructType, P4HIR::HeaderType, P4HIR::HeaderUnionType>(parentType)) {
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
    ConversionTracer trace("Converting ", str);

    auto type = getOrCreateType(str->structType);

    auto loc = getLoc(builder, str);
    llvm::SmallVector<mlir::Value, 4> fields;
    for (const auto *field : str->components) {
        fields.push_back(convert(field->expression));
    }

    // If this is header, make it valid as well
    if (mlir::isa<P4HIR::HeaderType>(type))
        fields.push_back(emitValidityConstant(loc, P4HIR::ValidityBit::Valid));

    setValue(str, builder.create<P4HIR::StructOp>(loc, type, fields).getResult());

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ListExpression *lst) {
    ConversionTracer trace("Converting ", lst);

    auto type = getOrCreateType(lst->type);

    auto loc = getLoc(builder, lst);
    llvm::SmallVector<mlir::Value, 4> fields;
    for (const auto *field : lst->components) {
        fields.push_back(convert(field));
    }

    setValue(lst, builder.create<P4HIR::TupleOp>(loc, type, fields).getResult());

    return false;
}

void P4HIRConverter::postorder(const P4::IR::ArrayIndex *arr) {
    ConversionTracer trace("Converting ", arr);

    auto lhs = getValue(arr->left);
    auto loc = getLoc(builder, arr);
    if (mlir::isa<mlir::TupleType>(lhs.getType())) {
        auto idx = mlir::cast<P4HIR::IntAttr>(getOrCreateConstantExpr(arr->right));
        setValue(arr, builder.create<P4HIR::TupleExtractOp>(loc, lhs, idx).getResult());
        return;
    }

    BUG("cannot handle this array yet: %1%", arr);
}

void P4HIRConverter::postorder(const P4::IR::Range *range) {
    ConversionTracer trace("Converting ", range);

    auto lhs = getValue(range->left, getIntType(range->left->type));
    auto rhs = getValue(range->right, getIntType(range->right->type));

    auto loc = getLoc(builder, range);
    setValue(range, builder.create<P4HIR::RangeOp>(loc, lhs, rhs).getResult());
}

void P4HIRConverter::postorder(const P4::IR::Mask *range) {
    ConversionTracer trace("Converting ", range);

    auto lhs = getValue(range->left, getIntType(range->left->type));
    auto rhs = getValue(range->right, getIntType(range->right->type));

    auto loc = getLoc(builder, range);
    setValue(range, builder.create<P4HIR::MaskOp>(loc, lhs, rhs).getResult());
}

bool P4HIRConverter::preorder(const P4::IR::P4Parser *parser) {
    ConversionTracer trace("Converting ", parser);
    ValueScope scope(p4Values);

    auto annotations = convert(parser->getAnnotations());

    auto applyType = mlir::cast<P4HIR::FuncType>(getOrCreateType(parser->getApplyMethodType()));
    auto ctorType =
        mlir::cast<P4HIR::CtorType>(getOrCreateConstructorType(parser->getConstructorMethodType()));
    auto argAttrs = convertParamAttributes(parser->getApplyParameters());
    assert(applyType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto loc = getLoc(builder, parser);
    auto parserOp = builder.create<P4HIR::ParserOp>(loc, parser->name.string_view(), applyType,
                                                    ctorType, argAttrs, annotations);
    parserOp.createEntryBlock();
    auto parserSymbol = mlir::StringAttr::get(builder.getContext(), parser->name.string_view());

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
        auto paramType = getOrCreateType(param->type);
        auto placeholder = P4HIR::CtorParamAttr::get(
            paramType, mlir::SymbolRefAttr::get(parserSymbol), builder.getStringAttr(paramName));
        auto val = builder.create<P4HIR::ConstOp>(getLoc(builder, param), placeholder, paramName);
        setValue(param, val);
    }

    // Materialize locals
    visit(parser->parserLocals);

    // Walk over all states, materializing the bodies
    visit(parser->states);

    // Create default transition (to start state)
    auto startStateSymbol =
        mlir::SymbolRefAttr::get(builder.getContext(), P4::IR::ParserState::start.string_view());

    builder.create<P4HIR::ParserTransitionOp>(
        getEndLoc(builder, parser), mlir::SymbolRefAttr::get(parserSymbol, {startStateSymbol}));

    auto [it, inserted] = p4Symbols.try_emplace(parser, mlir::SymbolRefAttr::get(parserOp));
    BUG_CHECK(inserted, "duplicate translation of %1%", parser);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::ParserState *state) {
    ConversionTracer trace("Converting ", state);
    ValueScope scope(p4Values);

    auto annotations = convert(state->annotations);

    auto stateOp = builder.create<P4HIR::ParserStateOp>(
        getLoc(builder, state), state->name.string_view(),
        annotations.empty() ? mlir::DictionaryAttr() : annotations);
    mlir::Block &first = stateOp.getBody().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&first);

    // Materialize all state components
    visit(state->components);

    const auto *parser = findContext<P4::IR::P4Parser>();
    auto parserSymbol = mlir::StringAttr::get(builder.getContext(), parser->name.string_view());

    // accept / reject states are special, their bodies contain only accept / reject ops
    if (state->name == P4::IR::ParserState::accept) {
        builder.create<P4HIR::ParserAcceptOp>(getLoc(builder, state));
        return false;
    } else if (state->name == P4::IR::ParserState::reject) {
        builder.create<P4HIR::ParserRejectOp>(getLoc(builder, state));
        return false;
    }

    // Normal transition is either PathExpression or SelectExpression
    if (const auto *pe = state->selectExpression->to<P4::IR::PathExpression>()) {
        LOG4("Resolving direct transition: " << pe);
        auto loc = getLoc(builder, pe);
        const auto *nextState = resolvePath(pe->path, false)->checkedTo<P4::IR::ParserState>();
        auto nextStateSymbol =
            mlir::SymbolRefAttr::get(builder.getContext(), nextState->name.string_view());
        builder.create<P4HIR::ParserTransitionOp>(
            loc, mlir::SymbolRefAttr::get(parserSymbol, {nextStateSymbol}));
    } else {
        LOG4("Resolving select transition: " << state->selectExpression);
        visit(state->selectExpression);
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::SelectExpression *select) {
    ConversionTracer trace("Converting ", select);

    const auto *parser = findContext<P4::IR::P4Parser>();
    auto parserSymbol = mlir::StringAttr::get(builder.getContext(), parser->name.string_view());

    // Materialize value to select over. Select is always a ListExpression,
    // even if it contains a single value. Lists ae lowered to tuples,
    // however, single value cases are not single-value tuples. Unwrap
    // single-value ListExpression down to the sole component.
    const P4::IR::Expression *selectArg = select->select;
    if (select->select->components.size() == 1) selectArg = select->select->components.front();

    // Create select itself
    auto transitionSelectOp = builder.create<P4HIR::ParserTransitionSelectOp>(
        getLoc(builder, select), convert(selectArg));
    mlir::Block &first = transitionSelectOp.getBody().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&first);

    bool hasDefaultCase = false;
    for (const auto *selectCase : select->selectCases) {
        const auto *nextState =
            resolvePath(selectCase->state->path, false)->checkedTo<P4::IR::ParserState>();
        auto nextStateSymbol =
            mlir::SymbolRefAttr::get(builder.getContext(), nextState->name.string_view());
        builder.create<P4HIR::ParserSelectCaseOp>(
            getLoc(builder, selectCase),
            [&](mlir::OpBuilder &b, mlir::Location) {
                const auto *keyset = selectCase->keyset;
                auto endLoc = getEndLoc(builder, keyset);
                mlir::Value keyVal;

                // Type inference does not do proper type unification for the key,
                // so we'd need to do this by ourselves
                auto convertElement = [&](const P4::IR::Expression *expr) -> mlir::Value {
                    // Universal set
                    if (expr->is<P4::IR::DefaultExpression>())
                        return b.create<P4HIR::UniversalSetOp>(endLoc).getResult();

                    auto elVal = convert(expr);
                    if (!mlir::isa<P4HIR::SetType>(elVal.getType()))
                        elVal = b.create<P4HIR::SetOp>(getEndLoc(builder, expr), elVal);
                    return elVal;
                };

                if (const auto *keyList = keyset->to<P4::IR::ListExpression>()) {
                    // Set product
                    llvm::SmallVector<mlir::Value, 4> elements;
                    for (const auto *element : keyList->components)
                        elements.push_back(convertElement(element));
                    // Treat product consisting entirely of universal sets as default case
                    hasDefaultCase |= llvm::all_of(elements, [](mlir::Value el) {
                        return mlir::isa<P4HIR::UniversalSetOp>(el.getDefiningOp());
                    });
                    keyVal = b.create<P4HIR::SetProductOp>(endLoc, elements);
                } else {
                    keyVal = convertElement(keyset);
                    hasDefaultCase |= mlir::isa<P4HIR::UniversalSetOp>(keyVal.getDefiningOp());
                }
                b.create<P4HIR::YieldOp>(endLoc, keyVal);
            },
            mlir::SymbolRefAttr::get(parserSymbol, {nextStateSymbol}));
    }

    // If there is no default case, then synthesize one explicitly
    // FIXME: signal `error.NoMatch` error.
    if (!hasDefaultCase) {
        auto rejectStateSymbol = mlir::SymbolRefAttr::get(
            builder.getContext(), P4::IR::ParserState::reject.string_view());

        auto endLoc = getEndLoc(builder, select);
        builder.create<P4HIR::ParserSelectCaseOp>(
            endLoc,
            [&](mlir::OpBuilder &b, mlir::Location) {
                b.create<P4HIR::YieldOp>(endLoc,
                                         builder.create<P4HIR::UniversalSetOp>(endLoc).getResult());
            },
            mlir::SymbolRefAttr::get(parserSymbol, {rejectStateSymbol}));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Declaration_Instance *decl) {
    ConversionTracer trace("Converting ", decl);

    auto annotations = convert(decl->annotations);
    if (annotations.empty()) annotations = mlir::DictionaryAttr();

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

    auto resultType = getOrCreateType(type);

    // Resolve to base type
    if (const auto *tdef = type->to<P4::IR::Type_Typedef>()) {
        type = resolveType(tdef->type);
        CHECK_NULL(type);
        LOG4("Resolved to typedef type: " << dbp(type));
    }
    if (const auto *spec = type->to<P4::IR::Type_Specialized>()) {
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
            // TODO: Handle @optional
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
    if (const auto *parser = type->to<P4::IR::P4Parser>()) {
        LOG4("resolved as parser instantiation");
        auto parserSym = p4Symbols.lookup(parser);
        BUG_CHECK(parserSym, "expected reference parser to be converted: %1%", dbp(parser));

        auto instance = builder.create<P4HIR::InstantiateOp>(getLoc(builder, decl), resultType,
                                                             parserSym.getRootReference(), operands,
                                                             decl->name.string_view(), annotations);
        setValue(decl, instance.getResult());
    } else if (const auto *ext = type->to<P4::IR::Type_Extern>()) {
        LOG4("resolved as extern instantiation");

        auto externName = builder.getStringAttr(ext->name.string_view());
        auto instance =
            builder.create<P4HIR::InstantiateOp>(getLoc(builder, decl), resultType, externName,
                                                 operands, decl->name.string_view(), annotations);
        setValue(decl, instance.getResult());
    } else if (const auto *control = type->to<P4::IR::P4Control>()) {
        LOG4("resolved as control instantiation");
        auto controlSym = p4Symbols.lookup(control);
        BUG_CHECK(controlSym, "expected reference control to be converted: %1%", dbp(control));

        auto instance = builder.create<P4HIR::InstantiateOp>(
            getLoc(builder, decl), resultType, controlSym.getRootReference(), operands,
            decl->name.string_view(), annotations);
        setValue(decl, instance.getResult());
    } else if (const auto *pkg = type->to<P4::IR::Type_Package>()) {
        LOG4("resolved as package instantiation");

        auto packageName = builder.getStringAttr(pkg->name.string_view());
        auto instance =
            builder.create<P4HIR::InstantiateOp>(getLoc(builder, decl), resultType, packageName,
                                                 operands, decl->name.string_view(), annotations);
        setValue(decl, instance.getResult());
    } else {
        BUG("unsupported instance: %1% (of type %2%)", decl, dbp(type));
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Type_Extern *ext) {
    auto loc = getLoc(builder, ext);

    // TODO: Move to common method
    llvm::SmallVector<mlir::Type> typeParameters;
    for (const auto *type : ext->getTypeParameters()->parameters) {
        typeParameters.push_back(getOrCreateType(type));
    }

    auto annotations = convert(ext->annotations);

    auto extOp = builder.create<P4HIR::ExternOp>(
        loc, ext->name.string_view(),
        typeParameters.empty() ? mlir::ArrayAttr() : builder.getTypeArrayAttr(typeParameters),
        annotations.empty() ? mlir::DictionaryAttr() : annotations);
    extOp.createEntryBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&extOp.getBody().front());

    // Materialize method declarations
    visit(ext->methods);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Type_Package *pkg) {
    auto loc = getLoc(builder, pkg);

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

            ovl = builder.create<P4HIR::OverloadSetOp>(loc, origSymName);
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

    builder.create<P4HIR::PackageOp>(loc, symName, ctorType, typeParameters, argAttrs, annotations);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Control *control) {
    ConversionTracer trace("Converting ", control);
    ValueScope scope(p4Values);

    auto annotations = convert(control->getAnnotations());

    auto applyType = mlir::cast<P4HIR::FuncType>(getOrCreateType(control->getApplyMethodType()));
    auto ctorType = mlir::cast<P4HIR::CtorType>(
        getOrCreateConstructorType(control->getConstructorMethodType()));

    auto argAttrs = convertParamAttributes(control->getApplyParameters());
    assert(applyType.getNumInputs() == argAttrs.size() && "invalid parameter conversion");

    auto loc = getLoc(builder, control);
    auto controlOp = builder.create<P4HIR::ControlOp>(loc, control->name.string_view(), applyType,
                                                      ctorType, argAttrs, annotations);
    controlOp.createEntryBlock();
    auto controlSymbol = mlir::StringAttr::get(builder.getContext(), control->name.string_view());

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
        auto paramType = getOrCreateType(param->type);
        auto placeholder = P4HIR::CtorParamAttr::get(
            paramType, mlir::SymbolRefAttr::get(controlSymbol), builder.getStringAttr(paramName));
        auto val = builder.create<P4HIR::ConstOp>(getLoc(builder, param), placeholder, paramName);
        setValue(param, val);
    }

    // Materialize locals
    visit(control->controlLocals);

    {
        ValueScope scope(p4Values);

        auto applyOp = builder.create<P4HIR::ControlApplyOp>(getLoc(builder, control->body));
        mlir::Block &first = applyOp.getBody().emplaceBlock();

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&first);

        // Materialize body
        visit(control->body);
    }

    auto [it, inserted] = p4Symbols.try_emplace(control, mlir::SymbolRefAttr::get(controlOp));
    BUG_CHECK(inserted, "duplicate translation of %1%", control);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::P4Table *table) {
    ConversionTracer trace("Converting ", table);

    auto annotations = convert(table->annotations);

    auto loc = getLoc(builder, table);
    auto tableOp = builder.create<P4HIR::TableOp>(
        loc, table->name.string_view(), annotations, [&](mlir::OpBuilder &, mlir::Location) {
            // Materialize all properties
            for (const auto *prop : table->properties->properties) visit(prop);
        });

    auto [it, inserted] = p4Symbols.try_emplace(table, mlir::SymbolRefAttr::get(tableOp));
    BUG_CHECK(inserted, "duplicate translation of %1%", table);

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::Property *prop) {
    ConversionTracer trace("Converting ", prop);

    auto annotations = convert(prop->annotations);

    auto loc = getLoc(builder, prop);
    if (0) {
    } else if (prop->name == P4::IR::TableProperties::actionsPropertyName) {
        builder.create<P4HIR::TableActionsOp>(
            loc, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                const auto *alist = prop->value->checkedTo<P4::IR::ActionList>();
                for (const auto *act : alist->actionList) {
                    ValueScope scope(controlPlaneValues);

                    auto aAnnotations = convert(act->annotations);

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
                        controlPlaneTypes.push_back(getOrCreateType(param->type));
                        auto annotations = convert(param->annotations);

                        llvm::SmallVector<mlir::NamedAttribute> paramAttrs = {
                            builder.getNamedAttr(P4HIR::FuncOp::getParamNameAttrName(),
                                                 builder.getStringAttr(param->name.string_view())),
                        };
                        if (!annotations.empty())
                            paramAttrs.emplace_back(builder.getNamedAttr(
                                P4HIR::FuncOp::getParamAnnotationAttrName(), annotations));
                        controlPlaneParamAttrs.emplace_back(builder.getDictionaryAttr(paramAttrs));
                    }
                    auto funcType = P4HIR::FuncType::get(context(), controlPlaneTypes);
                    const auto *action =
                        resolvePath(expr->method->checkedTo<P4::IR::PathExpression>()->path, false)
                            ->checkedTo<P4::IR::P4Action>();
                    auto actSym = p4Symbols.lookup(action);
                    BUG_CHECK(actSym, "expected reference action to be converted: %1%", action);

                    b.create<P4HIR::TableActionOp>(
                        getLoc(builder, expr), actSym, funcType, controlPlaneParamAttrs,
                        aAnnotations,
                        [&](mlir::OpBuilder &, mlir::Block::BlockArgListType args, mlir::Location) {
                            for (const auto arg : args) {
                                const auto *param = params[argCount + arg.getArgNumber()];
                                controlPlaneValues.insert(param, arg);
                            }

                            visit(expr);
                        });
                }
            });
    } else if (prop->name == P4::IR::TableProperties::keyPropertyName) {
        builder.create<P4HIR::TableKeyOp>(
            loc, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(p4Values);

                const auto *key = prop->value->checkedTo<P4::IR::Key>();
                for (const auto *kel : key->keyElements) {
                    auto kAnnotations = convert(kel->annotations);
                    auto kExpr = convert(kel->expression);
                    const auto *match_kind = resolvePath(kel->matchType->path, false)
                                                 ->checkedTo<P4::IR::Declaration_ID>();
                    b.create<P4HIR::TableKeyEntryOp>(
                        getLoc(b, kel), b.getStringAttr(match_kind->name.string_view()), kExpr,
                        kAnnotations);
                }
            });
    } else if (prop->name == P4::IR::TableProperties::defaultActionPropertyName) {
        builder.create<P4HIR::TableDefaultActionOp>(
            loc, annotations, [&](mlir::OpBuilder &b, mlir::Location) {
                ValueScope scope(p4Values);

                const auto *expr = prop->value->checkedTo<P4::IR::ExpressionValue>()->expression;
                visit(expr);
            });
    } else if (prop->name == P4::IR::TableProperties::entriesPropertyName) {
        BUG("cannot handle entries yet");
    } else if (prop->name == P4::IR::TableProperties::sizePropertyName) {
        const auto *expr = prop->value->checkedTo<P4::IR::ExpressionValue>()->expression;
        // Here property value might be a constructor argument. So we need to
        // see, if we have a placeholder for it
        mlir::TypedAttr size;
        if (auto val = getValue(expr, {}, /* unchecked */ true))
            size = mlir::cast<P4HIR::ConstOp>(val.getDefiningOp()).getValue();
        else
            size = getOrCreateConstantExpr(expr);

        builder.create<P4HIR::TableSizeOp>(
            loc, size, annotations.empty() ? mlir::DictionaryAttr() : annotations);
    } else {
        builder.create<P4HIR::TableEntryOp>(
            loc, builder.getStringAttr(prop->getName().string_view()), prop->isConstant,
            annotations, [&](mlir::OpBuilder &b, mlir::Type &resultType, mlir::Location) {
                ValueScope scope(p4Values);

                const auto *expr = prop->value->checkedTo<P4::IR::ExpressionValue>()->expression;
                auto val = convert(expr);
                resultType = val.getType();
                b.create<P4HIR::YieldOp>(getEndLoc(b, prop), val);
            });
    }

    return false;
}

bool P4HIRConverter::preorder(const P4::IR::SwitchStatement *sw) {
    ConversionTracer trace("Converting ", sw);

    auto cond = convert(sw->expression);

    builder.create<P4HIR::SwitchOp>(
        getLoc(builder, sw), cond, [&](mlir::OpBuilder &b, mlir::Location) {
            llvm::SmallVector<mlir::Attribute> cases;

            for (const auto *swCase : sw->cases) {
                if (swCase->label->to<P4::IR::DefaultExpression>()) {
                    if (!cases.empty())
                        builder.create<P4HIR::CaseOp>(
                            getLoc(b, swCase), b.getArrayAttr(cases),
                            cases.size() > 1 ? P4HIR::CaseOpKind::Anyof : P4HIR::CaseOpKind::Equal,
                            [&](mlir::OpBuilder &b, mlir::Location) {
                                b.create<P4HIR::YieldOp>(getEndLoc(builder, swCase));
                            });

                    builder.create<P4HIR::CaseOp>(
                        getLoc(b, swCase), b.getArrayAttr({}), P4HIR::CaseOpKind::Default,
                        [&](mlir::OpBuilder &b, mlir::Location) {
                            visit(swCase->statement);
                            b.create<P4HIR::YieldOp>(getEndLoc(builder, swCase));
                        });
                    cases.clear();
                } else {
                    // Handle special case: action run enum
                    if (sw->expression->type->is<P4::IR::Type_ActionEnum>()) {
                        const auto *path = swCase->label->checkedTo<P4::IR::PathExpression>();
                        cases.push_back(P4HIR::EnumFieldAttr::get(cond.getType(),
                                                                  path->path->name.string_view()));
                    } else
                        cases.push_back(getOrCreateConstantExpr(swCase->label));

                    if (swCase->statement) {
                        builder.create<P4HIR::CaseOp>(
                            getLoc(b, swCase), b.getArrayAttr(cases),
                            cases.size() > 1 ? P4HIR::CaseOpKind::Anyof : P4HIR::CaseOpKind::Equal,
                            [&](mlir::OpBuilder &b, mlir::Location) {
                                ValueScope scope(p4Values);

                                visit(swCase->statement);
                                b.create<P4HIR::YieldOp>(getEndLoc(builder, swCase));
                            });
                        cases.clear();
                    }
                }
            }

            b.create<P4HIR::YieldOp>(getEndLoc(builder, sw));
        });

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
