#ifndef INCLUDE_P4MLIR_P4C_TYPE_CONVERTER_H_
#define INCLUDE_P4MLIR_P4C_TYPE_CONVERTER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#pragma GCC diagnostic pop

#include "mlir/IR/Types.h"

namespace P4::P4MLIR {
class P4HIRConverter;

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
    bool preorder(const P4::IR::Type_Varbits *type) override;
    bool preorder(const P4::IR::Type_Boolean *type) override;
    bool preorder(const P4::IR::Type_String *type) override;
    bool preorder(const P4::IR::Type_Unknown *type) override;
    bool preorder(const P4::IR::Type_Dontcare *type) override;
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
    bool preorder(const P4::IR::Type_Array *h) override;
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
    bool preorder(const P4::IR::Type_Set *s) override;

    mlir::Type getType() const { return type; }
    bool setType(const P4::IR::Type *type, mlir::Type mlirType);
    mlir::Type convert(const P4::IR::Type *type);

 private:
    P4HIRConverter &converter;
    mlir::Type type = nullptr;
};

}  // namespace P4::P4MLIR

#endif  // INCLUDE_P4MLIR_P4C_TYPE_CONVERTER_H_"
