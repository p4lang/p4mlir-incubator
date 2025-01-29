#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"

static mlir::ParseResult parseActionType(mlir::AsmParser &p,
                                         llvm::SmallVector<P4::P4MLIR::P4HIR::ParamType> &params);

static void printActionType(mlir::AsmPrinter &p,
                            mlir::ArrayRef<P4::P4MLIR::P4HIR::ParamType> params);

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

static mlir::ParseResult parseActionType(mlir::AsmParser &p,
                                         llvm::SmallVector<P4::P4MLIR::P4HIR::ParamType> &params) {
    if (failed(p.parseLParen())) return mlir::failure();

    // `(` `)`
    if (succeeded(p.parseOptionalRParen())) return mlir::success();

    // (direction)? type
    // TODO: Share with FuncType parsing
    auto parseParamType = [&]() {
        mlir::Type type;

        if (auto maybeType = p.parseOptionalType(type); maybeType.has_value()) {
            if (!succeeded(*maybeType)) return mlir::failure();

            params.push_back(ParamType::get(type));
            return mlir::success();
        }

        auto maybeDirection = mlir::FieldParser<ParamDirection>::parse(p);
        if (!succeeded(maybeDirection)) return mlir::failure();

        if (p.parseType(type)) return mlir::failure();
        params.push_back(ParamType::get(type, maybeDirection.value()));
        return mlir::success();
    };

    // paramType (`,` paramType)*
    if (failed(parseParamType())) return mlir::failure();
    while (succeeded(p.parseOptionalComma())) {
        if (failed(parseParamType())) return mlir::failure();
    }

    return p.parseRParen();
}

static void printActionType(mlir::AsmPrinter &p,
                            mlir::ArrayRef<P4::P4MLIR::P4HIR::ParamType> params) {
    p << '(';
    llvm::interleaveComma(params, p, [&p](ParamType type) {
        auto dir = type.getDir();
        if (dir != ParamDirection::None) {
            p.printStrippedAttrOrType(dir);
            p << ' ';
        }
        p.printType(type.getParamType());
    });
    p << ')';
}

void P4HIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"  // NOLINT
        >();
}
