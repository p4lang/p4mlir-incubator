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

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
    mlir::APInt APValue;

    if (!mlir::isa<BitsType, InfIntType>(odsType)) {
        parser.emitError(parser.getCurrentLocation(), "expected integer type");
        return {};
    }

    // Consume the '<' symbol.
    if (parser.parseLess()) return {};

    if (auto type = mlir::dyn_cast<BitsType>(odsType)) {
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
    if (auto type = mlir::dyn_cast<BitsType>(getType())) {
        if (type.isSigned())
            printer << getSInt();
        else
            printer << getUInt();
    } else
        printer << getValue();

    printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              APInt value) {
    if (!mlir::isa<BitsType, InfIntType>(type)) {
        emitError() << "expected 'simple.int' type";
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
    P4HIR::EnumType type;
    if (p.parseLess() || p.parseKeyword(&field) || p.parseComma() ||
        p.parseCustomTypeWithFallback<P4HIR::EnumType>(type) || p.parseGreater())
        return {};

    return EnumFieldAttr::get(type, field);
}

void EnumFieldAttr::print(AsmPrinter &p) const {
    p << "<" << getField().getValue() << ", ";
    p.printType(getType());
    p << ">";
}

EnumFieldAttr EnumFieldAttr::get(mlir::Type type, StringAttr value) {
    EnumType enumType = llvm::dyn_cast<EnumType>(type);
    if (!enumType) return nullptr;

    // Check whether the provided value is a member of the enum type.
    if (!enumType.contains(value.getValue())) {
        //    emitError() << "enum value '" << value.getValue()
        //                   << "' is not a member of enum type " << enumType;
        return nullptr;
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
