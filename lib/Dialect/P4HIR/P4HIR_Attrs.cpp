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

    if (!mlir::isa<BitsType>(odsType)) return {};
    auto type = mlir::cast<BitsType>(odsType);

    // Consume the '<' symbol.
    if (parser.parseLess()) return {};

    // Fetch arbitrary precision integer value.
    if (type.isSigned()) {
        int64_t value;
        if (parser.parseInteger(value))
            parser.emitError(parser.getCurrentLocation(), "expected integer value");
        APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
        if (APValue.getSExtValue() != value)
            parser.emitError(parser.getCurrentLocation(),
                             "integer value too large for the given type");
    } else {
        uint64_t value;
        if (parser.parseInteger(value))
            parser.emitError(parser.getCurrentLocation(), "expected integer value");
        APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
        if (APValue.getZExtValue() != value)
            parser.emitError(parser.getCurrentLocation(),
                             "integer value too large for the given type");
    }

    // Consume the '>' symbol.
    if (parser.parseGreater()) return {};

    return IntAttr::get(type, APValue);
}

void IntAttr::print(AsmPrinter &printer) const {
    auto type = mlir::cast<BitsType>(getType());
    printer << '<';
    if (type.isSigned())
        printer << getSInt();
    else
        printer << getUInt();
    printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                              APInt value) {
    if (!mlir::isa<BitsType>(type)) {
        emitError() << "expected 'simple.int' type";
        return failure();
    }

    auto intType = mlir::cast<BitsType>(type);
    if (value.getBitWidth() != intType.getWidth()) {
        emitError() << "type and value bitwidth mismatch: " << intType.getWidth()
                    << " != " << value.getBitWidth();
        return failure();
    }

    return success();
}

void P4HIRDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.cpp.inc"  // NOLINT
        >();
}
