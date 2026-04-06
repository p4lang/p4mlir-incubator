#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "p4mlir/P4C/type_converter.h"

#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/P4C/translate.h"
#pragma GCC diagnostic pop

using namespace P4::P4MLIR;

namespace {
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

bool P4TypeConverter::preorder(const P4::IR::Type_Varbits *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::VarBitsType::get(converter.context(), type->size);
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

bool P4TypeConverter::preorder(const P4::IR::Type_Dontcare *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    auto mlirType = P4HIR::DontcareType::get(converter.context());
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

bool P4TypeConverter::preorder(const P4::IR::Type_Set *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);

    auto mlirType = P4HIR::SetType::get(convert(type->elementType));
    return setType(type, mlirType);
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

bool P4TypeConverter::preorder(const P4::IR::Type_Array *type) {
    if ((this->type = converter.findType(type))) return false;

    ConversionTracer trace("TypeConverting ", type);
    llvm::SmallVector<P4HIR::FieldInfo, 4> fields;

    auto sz = mlir::cast<P4HIR::IntAttr>(converter.getOrCreateConstantExpr(type->size)).getUInt();
    auto elementType = convert(type->elementType);
    mlir::Type mlirType;
    // Header stack are arrays over headers or header unions.
    if (mlir::isa<P4HIR::HeaderType, P4HIR::HeaderUnionType>(elementType))
        mlirType = P4HIR::HeaderStackType::get(
            converter.context(), sz, mlir::cast<P4HIR::StructLikeTypeInterface>(elementType));
    else
        mlirType = P4HIR::ArrayType::get(converter.context(), sz, elementType);

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
    auto mlirType = P4HIR::EnumType::get(converter.context(), cases);
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
        // Hack for handling dontcare type: if it is in aggregate, then it
        // should be really be !p4hir.set<!p4hir.dontcare> as this would
        // represent default expression (unfortunaltely p4c type inference is
        // broken here)
        auto fieldTy = convert(field);
        if (mlir::isa<P4HIR::DontcareType>(fieldTy)) fieldTy = P4HIR::SetType::get(fieldTy);
        fields.push_back(fieldTy);
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
