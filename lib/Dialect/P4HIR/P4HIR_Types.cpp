#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"

static mlir::ParseResult parseActionType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params);

static void printActionType(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params);

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;

void BitsType::print(mlir::AsmPrinter &printer) const {
    printer << (isSigned() ? "int" : "bit") << '<' << getWidth() << '>';
}

Type BitsType::parse(mlir::AsmParser &parser, bool isSigned) {
    auto *context = parser.getBuilder().getContext();

    if (parser.parseLess()) return {};

    // Fetch integer size.
    unsigned width;
    if (parser.parseInteger(width)) return {};

    if (parser.parseGreater()) return {};

    return BitsType::get(context, width, isSigned);
}

Type BoolType::parse(mlir::AsmParser &parser) { return get(parser.getContext()); }

void BoolType::print(mlir::AsmPrinter &printer) const {}

Type P4HIRDialect::parseType(mlir::DialectAsmParser &parser) const {
    SMLoc typeLoc = parser.getCurrentLocation();
    StringRef mnemonic;
    Type genType;

    // Try to parse as a tablegen'd type.
    OptionalParseResult parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if (parseResult.has_value()) return genType;

    // Type is not tablegen'd: try to parse as a raw C++ type.
    return StringSwitch<function_ref<Type()>>(mnemonic)
        .Case("int", [&] { return BitsType::parse(parser, /* isSigned */ true); })
        .Case("bit", [&] { return BitsType::parse(parser, /* isSigned */ false); })
        .Default([&] {
            parser.emitError(typeLoc) << "unknown P4HIR type: " << mnemonic;
            return Type();
        })();
}

void P4HIRDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &os) const {
    // Try to print as a tablegen'd type.
    if (generatedTypePrinter(type, os).succeeded()) return;

    // Add some special handling for certain types
    TypeSwitch<Type>(type).Case<BitsType>([&](BitsType type) { type.print(os); }).Default([](Type) {
        llvm::report_fatal_error("printer is missing a handler for this type");
    });
}

ActionType ActionType::clone(TypeRange inputs, TypeRange results) const {
    assert(results.size() == 0 && "expected exactly zero result type");
    return get(getContext(), llvm::to_vector(inputs));
}

static mlir::ParseResult parseActionType(mlir::AsmParser &p,
                                         llvm::SmallVector<mlir::Type> &params) {
    return p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        mlir::Type type;
        if (p.parseType(type)) return mlir::failure();
        params.push_back(type);
        return mlir::success();
    });

    return p.parseRParen();
}

static void printActionType(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params) {
    p << '(';
    llvm::interleaveComma(params, p, [&p](mlir::Type type) { p.printType(type); });
    p << ')';
}

void P4HIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"  // NOLINT
        >();
}
