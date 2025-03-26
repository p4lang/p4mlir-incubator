#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;

mlir::Type IntAttr::getType() const { return getImpl()->type; }

llvm::APInt IntAttr::getValue() const { return getImpl()->value; }

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
    mlir::APInt APValue;
    mlir::Type valType = odsType;

    while (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(valType))
        valType = aliasType.getAliasedType();

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
    while (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type))
        type = aliasType.getAliasedType();

    if (auto bitsType = mlir::dyn_cast<BitsType>(type)) {
        APInt val = getValue();
        val.print(printer.getStream(), bitsType.isSigned());
    } else
        printer << getValue();

    printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              APInt value) {
    while (auto aliasType = mlir::dyn_cast<AliasType>(type)) type = aliasType.getAliasedType();

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

    return EnumFieldAttr::get(type, field);
}

void ErrorCodeAttr::print(AsmPrinter &p) const {
    p << "<" << getField().getValue() << ", ";
    p.printType(getType());
    p << ">";
}

ErrorCodeAttr ErrorCodeAttr::get(mlir::Type type, StringAttr value) {
    ErrorType errorType = llvm::dyn_cast<ErrorType>(type);
    if (!errorType) return nullptr;

    // Check whether the provided value is a member of the enum type.
    if (!errorType.contains(value.getValue())) {
        //    emitError() << "error code '" << value.getValue()
        //                   << "' is not a member of error type " << errorType;
        return nullptr;
    }

    return Base::get(value.getContext(), type, value);
}

void P4HIRDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.cpp.inc"  // NOLINT
        >();
}
