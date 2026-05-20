#include "p4mlir/Dialect/P4HIR/P4HIR_Mangle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

static void getNameImpl(llvm::raw_ostream &os, mlir::Type type);

static void mangleTypes(llvm::raw_ostream &os, llvm::ArrayRef<mlir::Type> types) {
    for (auto type : types) getNameImpl(os, type);
}

static void mangleIdentifier(llvm::raw_ostream &os, llvm::StringRef str) {
    os << str.size() << str;
}

static void mangleIdentifier(llvm::raw_ostream &os, mlir::StringAttr str) {
    mangleIdentifier(os, str.strref());
}

static void mangleString(llvm::raw_ostream &os, llvm::StringRef str) {
    os << "S" << str.size() << "_" << str;
}

static void mangleInt(llvm::raw_ostream &os, P4HIR::IntAttr attr) {
    os << "I" << attr.getValue() << "_";
}

static void mangleAnnotations(llvm::raw_ostream &os, mlir::DictionaryAttr ann) { os << "A"; }

static void mangleFields(llvm::raw_ostream &os, llvm::ArrayRef<P4HIR::FieldInfo> fields) {
    for (auto field : fields) {
        // Do not mangle validity bit if any
        if (mlir::isa<P4HIR::ValidBitType>(field.type)) continue;

        mangleIdentifier(os, field.name);
        getNameImpl(os, field.type);
        if (field.annotations) mangleAnnotations(os, field.annotations);
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::AliasType type) {
    os << "n";
    getNameImpl(os, type.getAliasedType());
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ArrayType type) {
    os << "a" << type.getSize();
    getNameImpl(os, type.getElementType());
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::FuncType type) {
    os << "f";
    getNameImpl(os, type.getReturnType());
    os << "_";
    mangleTypes(os, type.getInputs());
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments());
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ParserType type) {
    os << "p";
    mangleIdentifier(os, type.getName());
    mangleTypes(os, type.getInputs());
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments());
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ControlType type) {
    os << "c";
    mangleIdentifier(os, type.getName());
    mangleTypes(os, type.getInputs());
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments());
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::EnumType type) {
    os << "ee";
    mangleIdentifier(os, type.getName());
    for (auto enumCase : type.getFields().getAsRange<mlir::StringAttr>()) {
        mangleIdentifier(os, enumCase);
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::SerEnumType type) {
    os << "es";
    mangleIdentifier(os, type.getName());
    getNameImpl(os, type.getType());
    for (auto enumCase : type.getFields()) {
        mangleIdentifier(os, enumCase.getName());
        mangleInt(os, cast<P4HIR::IntAttr>(enumCase.getValue()));
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ErrorType type) {
    os << "er";
    for (auto enumCase : type.getFields().getAsRange<mlir::StringAttr>()) {
        mangleIdentifier(os, enumCase);
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ExternType type) {
    os << "x";
    mangleIdentifier(os, type.getName());
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments());
    }
}

static void getNameImpl(llvm::raw_ostream &os, mlir::Type type) {
    llvm::TypeSwitch<mlir::Type>(type)
        .Case<P4HIR::ArrayType>([&](auto type) { /* a */
                                                 getNameImpl(os, type);
        })
        .Case<P4HIR::BoolType>([&](auto) { os << "b"; })
        .Case<P4HIR::ControlType>([&](auto type) { /* c */
                                                   getNameImpl(os, type);
        })
        .Case<P4HIR::DontcareType>([&](auto) { os << "d"; })
        .Case<P4HIR::EnumType>([&](auto type) { /* ee */
                                                getNameImpl(os, type);
        })
        .Case<P4HIR::SerEnumType>([&](auto type) { /* es */
                                                   getNameImpl(os, type);
        })
        .Case<P4HIR::ErrorType>([&](auto type) { /* er */
                                                 getNameImpl(os, type);
        })
        .Case<P4HIR::FuncType>([&](auto type) { /* f */
                                                getNameImpl(os, type);
        })
        // .Case<P4HIR::PackageType>([&](auto type) { os << "g";})
        .Case<P4HIR::HeaderType>([&](auto type) {
            os << "h";
            mangleIdentifier(os, type.getName());
            mangleFields(os, type.getFields());
        })
        .Case<P4HIR::BitsType>(
            [&](auto type) { os << "i" << (type.isSigned() ? "s" : "u") << type.getWidth(); })
        .Case<P4HIR::InfIntType>([&](auto type) { os << "ii"; })
        .Case<P4HIR::VarBitsType>([&](auto type) { os << "iv" << type.getMaxWidth(); })
        // .Case<P4HIR::CtorType>([&](auto type) { os << "j";})
        .Case<P4HIR::HeaderStackType>([&](auto type) {
            os << "k";
            mangleIdentifier(os, type.getName());
            os << type.getArraySize();
            mangleFields(os, type.getArrayElementType().getFields());
        })
        .Case<P4HIR::SetType>([&](auto type) {
            os << "ls";
            getNameImpl(os, type.getElementType());
        })
        .Case<TupleType>([&](auto type) {
            os << "lt" << type.getTypes().size();
            mangleTypes(os, type.getTypes());
        })
        /* m */
        .Case<P4HIR::AliasType>([&](auto type) { /* n */
                                                 getNameImpl(os, type);
        })
        /* o */
        .Case<P4HIR::ParserType>([&](auto type) { /* p */
                                                  getNameImpl(os, type);
        })
        .Case<P4HIR::StringType>([&](auto) { os << "q"; })
        .Case<P4HIR::ReferenceType>([&](auto type) {
            os << "r";
            getNameImpl(os, type.getObjectType());
        })
        .Case<P4HIR::StructType>([&](auto type) {
            os << "s";
            mangleIdentifier(os, type.getName());
            mangleFields(os, type.getFields());
        })
        .Case<P4HIR::TypeVarType>([&](auto type) {
            os << "t";
            mangleIdentifier(os, type.getName());
        })
        .Case<P4HIR::UnknownType>([&](auto) { os << "u"; })
        .Case<P4HIR::VoidType>([&](auto) { os << "v"; })
        .Case<P4HIR::HeaderUnionType>([&](auto type) {
            os << "w";
            mangleIdentifier(os, type.getName());
            mangleFields(os, type.getFields());
        })
        .Case<P4HIR::ExternType>([&](auto type) { /* x */
                                                  getNameImpl(os, type);
        })
        /* y */
        /* z */

        .Default([](auto type) {
            type.dump();
            llvm_unreachable("unhandled type in mangler");
        });

    if (auto annotatedType = mlir::dyn_cast<P4HIR::AnnotatedType>(type);
        annotatedType && annotatedType.getAnnotationsAttr()) {
        /* A */
        mangleAnnotations(os, annotatedType.getAnnotationsAttr());
    }
}

void P4HIR::Mangler::getName(llvm::raw_ostream &os, mlir::Type type) const {
    os << "$";
    getNameImpl(os, type);
}

void P4HIR::Mangler::getName(llvm::SmallVectorImpl<char> &buf, mlir::Type type) const {
    llvm::raw_svector_ostream os(buf);
    getName(os, type);
}

mlir::StringAttr P4HIR::Mangler::getName(mlir::Type type) const {
    llvm::SmallString<256> nameBuf;
    getName(nameBuf, type);
    return mlir::StringAttr::get(type.getContext(), nameBuf.str());
}
