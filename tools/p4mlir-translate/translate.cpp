#include "translate.h"

#include "frontends/p4/typeMap.h"
#include "ir/ir-generated.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/big_int.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
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

namespace P4::P4MLIR {

namespace {

// Converts P4 SourceLocation stored in 'node' into its MLIR counterpart
mlir::Location getLoc(mlir::OpBuilder &builder, const IR::Node *node) {
    CHECK_NULL(node);
    auto sourceInfo = node->getSourceInfo();
    if (!sourceInfo.isValid()) return mlir::UnknownLoc::get(builder.getContext());

    const auto &start = sourceInfo.getStart();

    return mlir::FileLineColLoc::get(
        builder.getStringAttr(sourceInfo.getSourceFile().string_view()), start.getLineNumber(),
        start.getColumnNumber());
}

llvm::APInt toAPInt(const P4::big_int &value, unsigned bitWidth) {
    std::vector<uint64_t> valueBits;
    // Export absolute value into 64-bit unsigned values, most significant bit last
    export_bits(value, std::back_inserter(valueBits), 64, false);

    llvm::APInt apValue(bitWidth, valueBits);
    if (value < 0) apValue.negate();

    return apValue;
}

class P4HIRConverter;
class P4TypeConverter;

// A dedicated converter for conversion of the P4 types to their destination
// representation.
class P4TypeConverter : public P4::Inspector {
 public:
    P4TypeConverter(P4HIRConverter &converter, const P4::TypeMap *typeMap)
        : converter(converter), typeMap(typeMap) {
        CHECK_NULL(typeMap);
    }

    profile_t init_apply(const IR::Node *node) override {
        BUG_CHECK(!type, "Type already converted");
        return Inspector::init_apply(node);
    }

    void end_apply(const IR::Node *) override { BUG_CHECK(type, "Type not converted"); }

    bool preorder(const IR::Node *node) override {
        BUG_CHECK(node->is<IR::Type>(), "Invalid node");
        return false;
    }

    bool preorder(const P4::IR::Type *type) override {
        ::P4::error("%1%: P4 type not yet supported.", type);
        return false;
    }

    bool preorder(const P4::IR::Type_Bits *type) override;
    bool preorder(const P4::IR::Type_Boolean *type) override;
    bool preorder(const P4::IR::Type_Unknown *type) override;
    bool preorder(const P4::IR::Type_Typedef *type) override {
        visit(typeMap->getTypeType(type, true));
        return false;
    }

    bool preorder(const P4::IR::Type_Name *name) override {
        visit(typeMap->getTypeType(name, true));
        return false;
    }

    mlir::Type getType() { return type; }
    bool setType(const P4::IR::Type *type, mlir::Type mlirType);

 private:
    P4HIRConverter &converter;
    const P4::TypeMap *typeMap;
    mlir::Type type = nullptr;
};

class P4HIRConverter : public Inspector {
    mlir::OpBuilder &builder;

    const P4::TypeMap *typeMap = nullptr;
    llvm::DenseMap<const P4::IR::Type *, mlir::Type> p4Types;
    llvm::DenseMap<const P4::IR::Expression *, mlir::TypedAttr> p4Constants;
    llvm::DenseMap<const P4::IR::Node *, mlir::Value> p4Values;

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
        P4TypeConverter cvt(*this, typeMap);
        type->apply(cvt);
        return cvt.getType();
    }

    mlir::TypedAttr getOrCreateConstant(const P4::IR::Expression *expr) {
        auto cst = p4Constants.lookup(expr);
        if (cst) return cst;

        visit(expr);

        cst = p4Constants.lookup(expr);
        BUG_CHECK(cst, "expected %1% to be converted as constant", expr);
        return cst;
    }

    mlir::MLIRContext *context() const { return builder.getContext(); }

    bool preorder(const P4::IR::Node *node) override {
        ::P4::error("%1%: P4 construct not yet supported.", node);
        return false;
    }

    bool preorder(const P4::IR::Type *type) override {
        P4TypeConverter cvt(*this, typeMap);
        type->apply(cvt);
        return false;
    }

    bool preorder(const IR::P4Program *) override { return true; }
    bool preorder(const IR::Constant *c) override { return !p4Constants.contains(c); }
    bool preorder(const IR::BoolLiteral *b) override { return !p4Constants.contains(b); }
    bool preorder(const IR::Declaration_Constant *decl) override {
        // We only should visit it once
        BUG_CHECK(!p4Values.contains(decl), "duplicate decl conversion %1%", decl);
        return true;
    }

    void postorder(const IR::Constant *cst) override;
    void postorder(const IR::BoolLiteral *b) override;
    void postorder(const IR::Declaration_Constant *decl) override;
};

bool P4TypeConverter::preorder(const P4::IR::Type_Bits *type) {
    if ((this->type = converter.findType(type))) return false;

    auto mlirType = P4HIR::BitsType::get(converter.context(), type->width_bits(), type->isSigned);
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Boolean *type) {
    if ((this->type = converter.findType(type))) return false;

    auto mlirType = P4HIR::BoolType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::preorder(const P4::IR::Type_Unknown *type) {
    if ((this->type = converter.findType(type))) return false;

    auto mlirType = P4HIR::UnknownType::get(converter.context());
    return setType(type, mlirType);
}

bool P4TypeConverter::setType(const P4::IR::Type *type, mlir::Type mlirType) {
    this->type = mlirType;
    converter.setType(type, mlirType);
    return false;
}

void P4HIRConverter::postorder(const IR::Declaration_Constant *decl) {
    auto type = getOrCreateType(decl->type);
    auto init = getOrCreateConstant(decl->initializer);
    auto loc = getLoc(builder, decl);

    auto val = builder.create<P4HIR::ConstOp>(loc, type, init);
    auto [it, inserted] = p4Values.try_emplace(decl, val);
    BUG_CHECK(inserted, "duplicate conversion of %1%", decl);
}

void P4HIRConverter::postorder(const IR::Constant *cst) {
    auto type = llvm::cast<P4HIR::BitsType>(getOrCreateType(cst->type));
    llvm::APInt value = toAPInt(cst->value, type.getWidth());

    auto [it, inserted] = p4Constants.try_emplace(cst, P4HIR::IntAttr::get(context(), type, value));
    BUG_CHECK(inserted, "duplicate conversion of %1%", cst);
}

void P4HIRConverter::postorder(const IR::BoolLiteral *b) {
    // FIXME: For some reason type inference uses `Type_Unknown` for BoolLiteral (sic!)
    // auto type = llvm::cast<P4HIR::BoolType>(getOrCreateType(b->type));
    auto type = P4HIR::BoolType::get(context());

    auto [it, inserted] =
        p4Constants.try_emplace(b, P4HIR::BoolAttr::get(context(), type, b->value));
    BUG_CHECK(inserted, "duplicate conversion of %1%", b);
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> toMLIR(mlir::MLIRContext &context,
                                         const P4::IR::P4Program *program,
                                         const P4::TypeMap *typeMap) {
    mlir::OpBuilder builder(&context);
    auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(moduleOp.getBody());

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
