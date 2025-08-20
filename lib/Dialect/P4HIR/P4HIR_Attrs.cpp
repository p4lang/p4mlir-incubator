#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;

std::optional<llvm::APSInt> P4::P4MLIR::P4HIR::getConstantInt(mlir::Attribute attr) {
    return mlir::TypeSwitch<mlir::Attribute, std::optional<llvm::APSInt>>(attr)
        .Case<IntAttr>([&](IntAttr intAttr) {
            mlir::Type type = intAttr.getType();
            bool isSigned = mlir::isa<InfIntType>(type) || mlir::cast<BitsType>(type).isSigned();
            return llvm::APSInt(intAttr.getValue(), !isSigned);
        })
        .Case<EnumFieldAttr>([&](EnumFieldAttr enumFieldAttr) -> std::optional<llvm::APSInt> {
            auto serEnumType = mlir::dyn_cast<SerEnumType>(enumFieldAttr.getType());
            if (!serEnumType) return std::nullopt;

            bool isSigned = serEnumType.getType().isSigned();
            llvm::APInt val = serEnumType.valueOf<IntAttr>(enumFieldAttr.getField()).getValue();
            return llvm::APSInt(val, !isSigned);
        })
        .Case<BoolAttr>([&](BoolAttr boolAttr) {
            llvm::APInt val(1, boolAttr.getValue() ? 1 : 0);
            return llvm::APSInt(val, true);
        })
        .Default([](mlir::Attribute) { return std::nullopt; });
}

mlir::TypedAttr P4::P4MLIR::P4HIR::foldConstantCast(mlir::Type destType, mlir::Attribute srcAttr) {
    if (auto destBitsType = mlir::dyn_cast<BitsType>(destType)) {
        if (auto srcCst = getConstantInt(srcAttr)) {
            unsigned destWidth = destBitsType.getWidth();
            llvm::APInt newVal = srcCst->extOrTrunc(destWidth);
            return IntAttr::get(destBitsType, newVal);
        }
    }

    return {};
}

mlir::Type IntAttr::getType() const { return getImpl()->type; }

llvm::APInt IntAttr::getValue() const { return getImpl()->value; }

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
    mlir::APInt APValue;
    mlir::Type valType = odsType;

    if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(valType))
        valType = aliasType.getCanonicalType();

    if (!mlir::isa<BitsType, InfIntType>(valType)) {
        parser.emitError(parser.getCurrentLocation(), "expected integer type");
        return {};
    }

    // Consume the '<' symbol.
    if (parser.parseLess()) return {};

    if (auto type = mlir::dyn_cast<BitsType>(valType)) {
        // Fetch arbitrary precision integer value.
        if (type.isSigned()) {
            mlir::APInt value;
            if (parser.parseInteger(value)) {
                parser.emitError(parser.getCurrentLocation(), "expected integer value");
                return {};
            }
            if (!value.isSignedIntN(type.getWidth())) {
                parser.emitError(parser.getCurrentLocation(),
                                 "integer value too large for the given type");
                return {};
            }
            APValue = value.sextOrTrunc(type.getWidth());
        } else {
            mlir::APInt value;
            if (parser.parseInteger(value)) {
                parser.emitError(parser.getCurrentLocation(), "expected integer value");
                return {};
            }
            if (!value.isIntN(type.getWidth())) {
                parser.emitError(parser.getCurrentLocation(),
                                 "integer value too large for the given type");
                return {};
            }
            APValue = value.zextOrTrunc(type.getWidth());
        }
    } else if (parser.parseInteger(APValue)) {
        parser.emitError(parser.getCurrentLocation(), "expected integer value");
        return {};
    }

    // Consume the '>' symbol.
    if (parser.parseGreater()) return {};

    return IntAttr::get(odsType, APValue);
}

void IntAttr::print(AsmPrinter &printer) const {
    printer << '<';
    auto type = getType();
    if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type))
        type = aliasType.getCanonicalType();

    if (auto bitsType = mlir::dyn_cast<BitsType>(type)) {
        APInt val = getValue();
        val.print(printer.getStream(), bitsType.isSigned());
    } else
        printer << getValue();

    printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              APInt value) {
    if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type))
        type = aliasType.getCanonicalType();

    if (!mlir::isa<BitsType, InfIntType>(type)) {
        emitError() << "expected integer type";
        return failure();
    }

    if (auto intType = mlir::dyn_cast<BitsType>(type)) {
        if (value.getBitWidth() != intType.getWidth()) {
            emitError() << "type and value bitwidth mismatch: " << intType.getWidth()
                        << " != " << value.getBitWidth();
            return failure();
        }
    }

    return success();
}

Attribute EnumFieldAttr::parse(AsmParser &p, Type) {
    StringRef field;
    mlir::Type type;
    if (p.parseLess() || p.parseKeyword(&field) || p.parseComma() || p.parseType(type) ||
        p.parseGreater())
        return {};

    if (!mlir::isa<P4HIR::EnumType, P4HIR::SerEnumType>(type)) {
        p.emitError(p.getCurrentLocation(),
                    "enum_field attribute could only be used for enum or ser_enum types");
        return {};
    }

    return EnumFieldAttr::get(type, field);
}

void EnumFieldAttr::print(AsmPrinter &p) const {
    p << "<" << getField().getValue() << ", ";
    p.printType(getType());
    p << ">";
}

EnumFieldAttr EnumFieldAttr::get(mlir::Type type, StringAttr value) {
    if (EnumType enumType = llvm::dyn_cast<EnumType>(type)) {
        // Check whether the provided value is a member of the enum type.
        if (!enumType.contains(value.getValue())) return nullptr;
    } else {
        auto serEnumType = llvm::cast<SerEnumType>(type);
        // Check whether the provided value is a member of the enum type.
        if (!serEnumType.contains(value.getValue())) return nullptr;
    }

    return Base::get(value.getContext(), type, value);
}

Attribute ErrorCodeAttr::parse(AsmParser &p, Type) {
    StringRef field;
    P4HIR::ErrorType type;
    if (p.parseLess() || p.parseKeyword(&field) || p.parseComma() ||
        p.parseCustomTypeWithFallback<P4HIR::ErrorType>(type) || p.parseGreater())
        return {};

    return ErrorCodeAttr::get(type, field);
}

void ErrorCodeAttr::print(AsmPrinter &p) const {
    p << "<" << getField().getValue() << ", ";
    p.printType(getType());
    p << ">";
}

ErrorCodeAttr ErrorCodeAttr::get(mlir::Type type, StringAttr value) {
    ErrorType errorType = llvm::dyn_cast<ErrorType>(type);
    if (!errorType) return nullptr;

    // Check whether the provided value is a member of the error type.
    if (!errorType.contains(value.getValue())) {
        //    emitError() << "error code '" << value.getValue()
        //                   << "' is not a member of error type " << errorType;
        return nullptr;
    }

    return Base::get(value.getContext(), type, value);
}

LogicalResult AggAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              ArrayAttr value) {
    if (!type || mlir::isa<NoneType>(type)) {
        emitError() << "p4hir.aggregate attribute must be typed";
        return failure();
    }

    auto checkSize = [&](size_t typeSize, size_t initializerSize) {
        if (initializerSize != typeSize) {
            emitError() << "expected " << typeSize
                        << " fields in initializer, but got: " << initializerSize;
            return failure();
        }

        return success();
    };

    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
        .Case<P4HIR::StructLikeTypeInterface>([&](auto structLike) {
            if (failed(checkSize(structLike.getFields().size(), value.size()))) return failure();

            for (auto [index, field] : llvm::enumerate(value.getValue())) {
                if (auto typedField = mlir::dyn_cast<mlir::TypedAttr>(field)) {
                    const auto &structField = structLike.getFields()[index];
                    if (typedField.getType() != structField.type) {
                        emitError()
                            << "aggregate initializer type for struct field '" << structField.name
                            << "' must match, expected: " << structField.type
                            << ", got: " << typedField.getType();
                        return failure();
                    }
                } else {
                    emitError() << "aggregate initializer must be typed: " << field;
                    return failure();
                }
            }
            return success();
        })
        .Case<P4HIR::ArrayType>([&](auto array) {
            if (failed(checkSize(array.getSize(), value.size()))) return failure();
            auto eltType = array.getElementType();
            for (auto field : value.getValue()) {
                if (auto typedField = mlir::dyn_cast<mlir::TypedAttr>(field)) {
                    if (typedField.getType() != eltType) {
                        emitError()
                            << "aggregate initializer type for array element must match, expected: "
                            << eltType << ", got: " << typedField.getType();
                        return failure();
                    }
                } else {
                    emitError() << "aggregate initializer must be typed: " << field;
                    return failure();
                }
            }
            return success();
        })
        .Case<mlir::TupleType>([&](auto tuple) {
            if (failed(checkSize(tuple.size(), value.size()))) return failure();
            for (auto [index, field] : llvm::enumerate(value.getValue())) {
                if (auto typedField = mlir::dyn_cast<mlir::TypedAttr>(field)) {
                    const auto &tupleFieldType = tuple.getType(index);
                    if (typedField.getType() != tupleFieldType) {
                        emitError() << "aggregate initializer type for tuple field " << index
                                    << " must match, expected: " << tupleFieldType
                                    << ", got: " << typedField.getType();
                        return failure();
                    }
                } else {
                    emitError() << "aggregate initializer must be typed: " << field;
                    return failure();
                }
            }
            return success();
        })
        .Default([&](auto) {
            emitError() << "expected aggregate type: " << type;
            return failure();
        });
}

mlir::Type UniversalSetAttr::getType() {
    return P4HIR::SetType::get(P4HIR::DontcareType::get(getContext()));
}

LogicalResult SetAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              P4HIR::SetKind kind, ArrayAttr value) {
    if (!type || mlir::isa<NoneType>(type)) {
        emitError() << "p4hir.set attribute must be typed";
        return failure();
    }

    auto setType = mlir::dyn_cast<P4HIR::SetType>(type);
    if (!setType) {
        emitError() << "p4hir.set attribute must have set type";
        return failure();
    }

    auto eltType = setType.getElementType();
    switch (kind) {
        case SetKind::Constant:
            // Check that all of constants are of the same type
            if (llvm::any_of(value.getAsRange<mlir::TypedAttr>(),
                             [&](auto attr) { return attr.getType() != eltType; })) {
                emitError() << "p4hir.set elements must be of the same type: " << eltType;
                return failure();
            }
            break;
        case SetKind::Range:
        case SetKind::Mask:
            if (value.size() != 2) {
                emitError() << "p4hir.set mask / range attribute must have only two values";
                return failure();
            }

            // Check that all of constants are of the same type
            if (llvm::any_of(value.getAsRange<mlir::TypedAttr>(),
                             [&](auto attr) { return attr.getType() != eltType; })) {
                emitError() << "p4hir.set elements must be of the same type: " << eltType;
                return failure();
            }

            break;
    }

    return success();
}

void P4HIRDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.cpp.inc"  // NOLINT
        >();
}
