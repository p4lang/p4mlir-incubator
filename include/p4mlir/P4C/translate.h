// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDE_P4MLIR_P4C_TRANSLATE_H_
#define INCLUDE_P4MLIR_P4C_TRANSLATE_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "ir/visitor.h"
#pragma GCC diagnostic pop

#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

namespace P4 {
namespace IR {
class P4Program;
}  // namespace IR
class BuiltInMethod;
class TypeMap;
}  // namespace P4

namespace P4::P4MLIR {

class P4HIRConverter : public P4::Inspector, public P4::ResolutionContext {
    mlir::OpBuilder &builder;

    P4::TypeMap *typeMap = nullptr;
    llvm::DenseMap<const P4::IR::Type *, mlir::Type> p4Types;
    // TODO: Implement unified constant map
    // using CTVOrExpr = std::variant<const P4::IR::CompileTimeValue *,
    //                                const P4::IR::Expression *>;
    // llvm::DenseMap<CTVOrExpr, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Expression *, mlir::TypedAttr> p4Constants;

    llvm::DenseMap<const P4::IR::P4Table *, llvm::SmallVector<mlir::Value>> tableKeyArgsMap;

    using ValueTable = llvm::ScopedHashTable<const P4::IR::Node *, mlir::Value>;
    // We temporary swap value table inside function / action to ensure everything
    // is properly isolated
    ValueTable *p4Values;
    ValueTable controlPlaneValues;
    using ValueScope = ValueTable::ScopeTy;

    using P4Symbol = std::variant<const P4::IR::Declaration *, const P4::IR::P4Parser *,
                                  const P4::IR::P4Control *>;
    using SymbolTable = llvm::ScopedHashTable<P4Symbol, mlir::SymbolRefAttr>;
    using SymbolScope = SymbolTable::ScopeTy;
    SymbolTable p4Symbols;

    bool defaultInitialize = false;

    mlir::TypedAttr resolveConstant(const P4::IR::CompileTimeValue *ctv);
    mlir::Value resolveReference(const P4::IR::Node *node, bool unchecked = false);

    mlir::Value getBoolConstant(mlir::Location loc, bool value) {
        return P4HIR::ConstOp::create(builder, loc, P4HIR::BoolAttr::get(context(), value));
    }
    mlir::Value getStringConstant(mlir::Location loc, llvm::Twine &bytes) {
        return P4HIR::ConstOp::create(
            builder, loc, mlir::StringAttr::get(bytes, P4HIR::StringType::get(context())));
    }
    mlir::Value getIntConstant(mlir::Location loc, llvm::APInt value, mlir::Type type) {
        return P4HIR::ConstOp::create(builder, loc, P4HIR::IntAttr::get(context(), type, value));
    }
    mlir::Value getUIntConstant(mlir::Location loc, uint64_t value, unsigned bitWidth) {
        return P4HIR::ConstOp::create(
            builder, loc,
            P4HIR::IntAttr::get(context(), P4HIR::BitsType::get(context(), bitWidth, false),
                                llvm::APInt(bitWidth, value)));
    }
    mlir::Value getSIntConstant(mlir::Location loc, uint64_t value, unsigned bitWidth) {
        return P4HIR::ConstOp::create(
            builder, loc,
            P4HIR::IntAttr::get(context(), P4HIR::BitsType::get(context(), bitWidth, true),
                                llvm::APInt(bitWidth, value, true)));
    }
    mlir::Value getUniversalSetConstant(mlir::Location loc) {
        return P4HIR::ConstOp::create(builder, loc, P4HIR::UniversalSetAttr::get(context()));
    }

    mlir::TypedAttr getTypedConstant(mlir::Type type, mlir::Attribute constant) {
        if (mlir::isa<P4HIR::BoolType>(type)) return mlir::cast<P4HIR::BoolAttr>(constant);

        if (mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(type))
            return mlir::cast<P4HIR::IntAttr>(constant);

        if (mlir::isa<P4HIR::ErrorType>(type)) return mlir::cast<P4HIR::ErrorCodeAttr>(constant);

        return mlir::cast<P4HIR::AggAttr>(constant);
    }

    P4HIR::BitsType getB32Type() { return P4HIR::BitsType::get(context(), 32, false); }

 public:
    P4HIRConverter(mlir::OpBuilder &builder, P4::TypeMap *typeMap, bool defaultInitialize = false)
        : builder(builder), typeMap(typeMap), defaultInitialize(defaultInitialize) {
        CHECK_NULL(typeMap);
    }

    mlir::Type findType(const P4::IR::Type *type) const { return p4Types.lookup(type); }

    mlir::Type setType(const P4::IR::Type *type, mlir::Type mlirType) {
        auto [it, inserted] = p4Types.try_emplace(type, mlirType);
        BUG_CHECK(inserted, "duplicate conversion for %1%", type);

        return it->second;
    }

    mlir::Type getOrCreateConstructorType(const P4::IR::Type_Method *type);

    virtual mlir::Type getOrCreateType(const P4::IR::Type *type);
    virtual mlir::Type getOrCreateType(const P4::IR::Expression *expr);

    virtual mlir::Type getOrCreateType(const P4::IR::Declaration_Variable *decl) {
        return P4HIR::ReferenceType::get(getOrCreateType(decl->type));
    }

    virtual mlir::Type getOrCreateType(const P4::IR::Parameter *param) {
        auto declType = getOrCreateType(param->type);
        return param->hasOut() ? P4HIR::ReferenceType::get(declType) : declType;
    }

    // Returns underlying type in case of something of serialized enum cate
    mlir::Type getIntType(const P4::IR::Type *type) {
        auto baseType = getOrCreateType(type);
        if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(baseType))
            baseType = aliasType.getCanonicalType();
        if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(baseType))
            baseType = serEnumType.getType();
        return baseType;
    }

    mlir::Value materializeConstantExpr(const P4::IR::Expression *expr);
    mlir::Value materializeConstantDecl(const P4::IR::Declaration_Constant *decl);

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

    mlir::Value getValueForSymbol(const P4::IR::Node *node, bool unchecked = false);
    mlir::Value getValue(const P4::IR::Node *node, mlir::Type type = {}, bool unchecked = false);
    mlir::Value getValue(mlir::Value val, mlir::Type type = {});
    mlir::Value setValue(const P4::IR::Node *node, mlir::Value value);

    mlir::Value convert(const P4::IR::Node *node) {
        visit(node);
        return getValue(node);
    }

    mlir::SymbolRefAttr setSymbol(P4Symbol symb, mlir::SymbolRefAttr value);

    /// Returns fully qualified symbols, if we're nested inside parser or control
    mlir::SymbolRefAttr getQualifiedSymbolRef(mlir::Operation *op);
    mlir::SymbolRefAttr getQualifiedSymbolRef(llvm::StringRef value) {
        return getQualifiedSymbolRef(builder.getStringAttr(value));
    }
    mlir::SymbolRefAttr getQualifiedSymbolRef(mlir::StringAttr attr);

    mlir::Attribute convertAnnotationExpr(const P4::IR::Expression *ann);
    mlir::Attribute convert(const P4::IR::Annotation *anns);
    mlir::DictionaryAttr convert(const P4::IR::Vector<P4::IR::Annotation> &ann);
    llvm::SmallVector<mlir::DictionaryAttr, 4> convertParamAttributes(
        const P4::IR::ParameterList *params);

    mlir::MLIRContext *context() const { return builder.getContext(); }
    mlir::OpBuilder &getBuilder() { return builder; }

    bool preorder(const P4::IR::Node *node) override {
        ::P4::error("P4 construct not yet supported: %1% (aka %2%)", node, dbp(node));
        return false;
    }

    bool preorder(const P4::IR::Type *type) override;
    bool preorder(const P4::IR::P4Program *p) override;
    bool preorder(const P4::IR::P4Action *a) override;
    bool preorder(const P4::IR::Function *f) override;

    bool preorder(const P4::IR::P4Parser *a) override;
    bool preorder(const P4::IR::ParserState *s) override;
    bool preorder(const P4::IR::SelectExpression *s) override;

    bool preorder(const P4::IR::Type_Extern *e) override;

    bool preorder(const P4::IR::P4Control *c) override;
    bool preorder(const P4::IR::P4Table *t) override;
    bool preorder(const P4::IR::Property *p) override;
    bool preorder(const P4::IR::ActionListElement *act) override;
    bool preorder(const P4::IR::Entry *ent) override;

    bool preorder(const P4::IR::Type_Package *e) override;

    bool preorder(const P4::IR::Method *m) override;
    bool preorder(const P4::IR::BlockStatement *block) override;
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
        // Should be resolved eslewhere, except for the constants
        if (const auto *cst = resolvePath(e->path, false)->to<P4::IR::Declaration_Constant>()) {
            setValue(e, materializeConstantDecl(cst));
            // visitAgain();
        }

        return false;
    }
    bool preorder(const P4::IR::InvalidHeader *h) override {
        materializeConstantExpr(h);
        visitAgain();
        return false;
    }
    bool preorder(const P4::IR::InvalidHeaderUnion *hu) override {
        materializeConstantExpr(hu);
        visitAgain();
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
    HANDLE_IN_POSTORDER(ContinueStatement)
    HANDLE_IN_POSTORDER(BreakStatement)
    HANDLE_IN_POSTORDER(ExitStatement)
    HANDLE_IN_POSTORDER(ArrayIndex)
    HANDLE_IN_POSTORDER(Range)
    HANDLE_IN_POSTORDER(Mask)

#undef HANDLE_IN_POSTORDER

#define HANDLE_IN_PREORDER(Node, Kind)                                \
    bool preorder(const P4::IR::Node *opAssign) override {            \
        return expandOpAssignBinOp(opAssign, P4HIR::BinOpKind::Kind); \
    }

    HANDLE_IN_PREORDER(MulAssign, Mul)
    HANDLE_IN_PREORDER(DivAssign, Div)
    HANDLE_IN_PREORDER(ModAssign, Mod)
    HANDLE_IN_PREORDER(AddAssign, Add)
    HANDLE_IN_PREORDER(SubAssign, Sub)
    HANDLE_IN_PREORDER(AddSatAssign, AddSat)
    HANDLE_IN_PREORDER(SubSatAssign, SubSat)
    HANDLE_IN_PREORDER(BAndAssign, And)
    HANDLE_IN_PREORDER(BOrAssign, Or)
    HANDLE_IN_PREORDER(BXorAssign, Xor)

#undef HANDLE_IN_PREORDER

#define HANDLE_IN_PREORDER(Node, ShiftOp)                     \
    bool preorder(const P4::IR::Node *opAssign) override {    \
        return expandOpAssignShift<P4HIR::ShiftOp>(opAssign); \
    }

    HANDLE_IN_PREORDER(ShlAssign, ShlOp)
    HANDLE_IN_PREORDER(ShrAssign, ShrOp)

#undef HANDLE_IN_PREORDER

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
    bool preorder(const P4::IR::ForStatement *fstmt) override;
    bool preorder(const P4::IR::ForInStatement *forin) override;
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
    mlir::Value emitValidityConstant(mlir::Location loc, P4HIR::ValidityBit validityConstValue);
    void emitHeaderValidityBitAssignOp(mlir::Location loc, mlir::Value header,
                                       P4HIR::ValidityBit validityConstValue);
    P4HIR::CmpOp emitHeaderIsValidCmpOp(mlir::Location loc, mlir::Value header,
                                        P4HIR::ValidityBit compareWith);
    P4HIR::CmpOp emitHeaderUnionIsValidCmpOp(mlir::Location loc, mlir::Value headerUnion,
                                             P4HIR::ValidityBit compareWith);
    void emitSetInvalidForAllHeaders(mlir::Location loc, mlir::Value headerUnion,
                                     const P4::cstring headerNameToSkip = nullptr);

    mlir::Value emitInvalidHeaderCmpOp(const P4::IR::Operation_Relation *p4RelationOp);
    mlir::Value emitInvalidHeaderUnionCmpOp(const P4::IR::Operation_Relation *p4RelationOp);
    mlir::Value emitHeaderBuiltInMethod(mlir::Location loc, const P4::BuiltInMethod *builtInMethod);
    mlir::Value emitHeaderUnionBuiltInMethod(mlir::Location loc,
                                             const P4::BuiltInMethod *builtInMethod);
    mlir::Value emitHeaderStackBuiltInMethod(mlir::Location loc,
                                             const P4::BuiltInMethod *builtInMethod);
    mlir::Type getObjectType(mlir::Value value) {
        if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(value.getType()))
            return refType.getObjectType();
        return value.getType();
    }
    bool expandOpAssignBinOp(const P4::IR::OpAssignmentStatement *opAssign, P4HIR::BinOpKind kind);

    template <typename ShiftOp>
    bool expandOpAssignShift(const P4::IR::OpAssignmentStatement *opAssign);
};

mlir::OwningOpRef<mlir::ModuleOp> toMLIR(mlir::MLIRContext &context,
                                         const P4::IR::P4Program *program, P4::TypeMap *typeMap);
mlir::OwningOpRef<mlir::ModuleOp> toMLIR(P4HIRConverter &conv, const P4::IR::P4Program *program);

}  // namespace P4::P4MLIR

#endif  // INCLUDE_P4MLIR_P4C_TRANSLATE_H_
