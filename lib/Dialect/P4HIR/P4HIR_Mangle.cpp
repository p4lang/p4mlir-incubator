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

struct ManglingContext {
    // Identifier substitutions.
    llvm::StringMap<unsigned> stringSubstitutions;
    unsigned nextSubstitutionIndex = 0;
    // Different mangling modes
    bool mangleFieldNames = false;
    bool mangleAnnotations = false;
};

static void getNameImpl(llvm::raw_ostream &os, mlir::Type type, ManglingContext &ctx);

static void mangleString(llvm::raw_ostream &os, llvm::StringRef str) {
    os << "S" << str.size() << "_" << str;
}

static void mangleInt(llvm::raw_ostream &os, const APInt &val) {
    os << "I";
    if (!val.isZero()) os << val - 1;
    os << "_";
}

static void mangleInt(llvm::raw_ostream &os, P4HIR::IntAttr attr) {
    mangleInt(os, attr.getValue());
}

static void mangleSubstitution(llvm::raw_ostream &os, unsigned idx) {
    os << 'Z';
    if (idx >= 26) {
        idx -= 26;
        if (idx) os << idx - 1;
        os << "_";
    } else {
        char substChar = idx + 'A';
        os << substChar;
    }
}

static void mangleIdentifier(llvm::raw_ostream &os, llvm::StringRef str, ManglingContext &ctx) {
    auto [iter, inserted] = ctx.stringSubstitutions.try_emplace(str, ctx.nextSubstitutionIndex);
    if (inserted) {
        ctx.nextSubstitutionIndex += 1;
        os << str.size() << str;
    } else {
        mangleSubstitution(os, iter->second);
    }
}

static void mangleIdentifier(llvm::raw_ostream &os, mlir::StringAttr str, ManglingContext &ctx) {
    mangleIdentifier(os, str.strref(), ctx);
}

static void mangleAnnotations(llvm::raw_ostream &os, mlir::DictionaryAttr ann) {
    if (ann.empty()) return;

    os << "A";

    // TBD
}

static void mangleTypes(llvm::raw_ostream &os, llvm::ArrayRef<mlir::Type> types,
                        ManglingContext &ctx) {
    for (auto type : types) getNameImpl(os, type, ctx);
}

static void mangleFields(llvm::raw_ostream &os, llvm::ArrayRef<P4HIR::FieldInfo> fields,
                         ManglingContext &ctx) {
    for (auto field : fields) {
        // Do not mangle validity bit if any
        if (mlir::isa<P4HIR::ValidBitType>(field.type)) continue;

        if (ctx.mangleFieldNames) mangleIdentifier(os, field.name, ctx);
        getNameImpl(os, field.type, ctx);
        if (ctx.mangleAnnotations && field.annotations) mangleAnnotations(os, field.annotations);
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::AliasType type, ManglingContext &ctx) {
    os << "n";
    getNameImpl(os, type.getAliasedType(), ctx);
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ArrayType type, ManglingContext &ctx) {
    os << "a" << type.getSize();
    getNameImpl(os, type.getElementType(), ctx);
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::FuncType type, ManglingContext &ctx) {
    os << "f";
    getNameImpl(os, type.getReturnType(), ctx);
    os << "_";
    mangleTypes(os, type.getInputs(), ctx);
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments(), ctx);
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ParserType type, ManglingContext &ctx) {
    os << "p";
    mangleIdentifier(os, type.getName(), ctx);
    mangleTypes(os, type.getInputs(), ctx);
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments(), ctx);
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ControlType type, ManglingContext &ctx) {
    os << "c";
    mangleIdentifier(os, type.getName(), ctx);
    mangleTypes(os, type.getInputs(), ctx);
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments(), ctx);
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::EnumType type, ManglingContext &ctx) {
    os << "ee";
    mangleIdentifier(os, type.getName(), ctx);
    if (ctx.mangleFieldNames) {
        for (auto enumCase : type.getFields().getAsRange<mlir::StringAttr>()) {
            mangleIdentifier(os, enumCase, ctx);
        }
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::SerEnumType type, ManglingContext &ctx) {
    os << "es";
    mangleIdentifier(os, type.getName(), ctx);
    getNameImpl(os, type.getType(), ctx);
    if (ctx.mangleFieldNames) {
        for (auto enumCase : type.getFields()) {
            mangleIdentifier(os, enumCase.getName(), ctx);
            mangleInt(os, cast<P4HIR::IntAttr>(enumCase.getValue()));
        }
    }
    os << "_";
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ErrorType type, ManglingContext &ctx) {
    os << "er";
    if (ctx.mangleFieldNames) {
        for (auto errorCase : type.getFields().getAsRange<mlir::StringAttr>())
            mangleIdentifier(os, errorCase, ctx);
        os << "_";
    }
}

static void getNameImpl(llvm::raw_ostream &os, P4HIR::ExternType type, ManglingContext &ctx) {
    os << "x";
    mangleIdentifier(os, type.getName(), ctx);
    if (type.getTypeArguments().size()) {
        os << "_";
        mangleTypes(os, type.getTypeArguments(), ctx);
    }
}

static void getNameImpl(llvm::raw_ostream &os, mlir::Type type, ManglingContext &ctx) {
    llvm::TypeSwitch<mlir::Type>(type)
        .Case<P4HIR::ArrayType>([&](auto type) { /* a */
                                                 getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::BoolType>([&](auto) { os << "b"; })
        .Case<P4HIR::ControlType>([&](auto type) { /* c */
                                                   getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::DontcareType>([&](auto) { os << "d"; })
        .Case<P4HIR::EnumType>([&](auto type) { /* ee */
                                                getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::SerEnumType>([&](auto type) { /* es */
                                                   getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::ErrorType>([&](auto type) { /* er */
                                                 getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::FuncType>([&](auto type) { /* f */
                                                getNameImpl(os, type, ctx);
        })
        // .Case<P4HIR::PackageType>([&](auto type) { os << "g";})
        .Case<P4HIR::HeaderType>([&](auto type) {
            os << "h";
            mangleIdentifier(os, type.getName(), ctx);
            mangleFields(os, type.getFields(), ctx);
        })
        .Case<P4HIR::BitsType>(
            [&](auto type) { os << (type.isSigned() ? "i" : "u") << type.getWidth(); })
        .Case<P4HIR::InfIntType>([&](auto) { os << "ii"; })
        .Case<P4HIR::VarBitsType>([&](auto type) { os << "iv" << type.getMaxWidth(); })
        // .Case<P4HIR::CtorType>([&](auto type) { os << "j";})
        .Case<P4HIR::HeaderStackType>([&](auto type) {
            os << "k";
            mangleIdentifier(os, type.getName(), ctx);
            os << type.getArraySize();
            mangleFields(os, type.getArrayElementType().getFields(), ctx);
        })
        .Case<P4HIR::SetType>([&](auto type) {
            os << "ls";
            getNameImpl(os, type.getElementType(), ctx);
        })
        .Case<TupleType>([&](auto type) {
            os << "lt" << type.getTypes().size();
            mangleTypes(os, type.getTypes(), ctx);
        })
        /* m  - free */
        .Case<P4HIR::AliasType>([&](auto type) { /* n */
                                                 getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::UnknownType>([&](auto) { os << "o"; })
        .Case<P4HIR::ParserType>([&](auto type) { /* p */
                                                  getNameImpl(os, type, ctx);
        })
        .Case<P4HIR::StringType>([&](auto) { os << "q"; })
        .Case<P4HIR::ReferenceType>([&](auto type) {
            os << "r";
            getNameImpl(os, type.getObjectType(), ctx);
        })
        .Case<P4HIR::StructType>([&](auto type) {
            os << "s";
            mangleIdentifier(os, type.getName(), ctx);
            mangleFields(os, type.getFields(), ctx);
        })
        .Case<P4HIR::TypeVarType>([&](auto type) {
            os << "t";
            mangleIdentifier(os, type.getName(), ctx);
        })
        /* u - bits type */
        .Case<P4HIR::VoidType>([&](auto) { os << "v"; })
        .Case<P4HIR::HeaderUnionType>([&](auto type) {
            os << "w";
            mangleIdentifier(os, type.getName(), ctx);
            mangleFields(os, type.getFields(), ctx);
        })
        .Case<P4HIR::ExternType>([&](auto type) { /* x */
                                                  getNameImpl(os, type, ctx);
        })
        /* y - free */
        /* z - free */

        .Default([](auto type) {
            type.dump();
            llvm_unreachable("unhandled type in mangler");
        });

    if (auto annotatedType = mlir::dyn_cast<P4HIR::AnnotatedType>(type);
        ctx.mangleAnnotations && annotatedType && annotatedType.getAnnotationsAttr()) {
        /* A */
        mangleAnnotations(os, annotatedType.getAnnotationsAttr());
    }
}

void P4HIR::Mangler::getName(llvm::raw_ostream &os, mlir::Type type) const {
    ManglingContext context;

    os << "$";
    getNameImpl(os, type, context);
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
