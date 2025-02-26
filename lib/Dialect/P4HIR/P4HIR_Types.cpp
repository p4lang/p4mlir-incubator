#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;
using namespace P4::P4MLIR::P4HIR::detail;

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params,
                                       mlir::Type &optionalResultType);
static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params);

static void printFuncType(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params,
                          mlir::Type optionalResultType = {});

#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"

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

FuncType FuncType::clone(TypeRange inputs, TypeRange results) const {
    assert(results.size() == 1 && "expected exactly one result type");
    return get(llvm::to_vector(inputs), results[0]);
}

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params) {
    mlir::Type placeholder;
    return parseFuncType(p, params, placeholder);
}

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params,
                                       mlir::Type &optionalReturnType) {
    // Parse return type, if any
    if (succeeded(p.parseOptionalLParen())) {
        // If we have already a '(', the function has no return type
        optionalReturnType = {};
    } else {
        mlir::Type type;
        if (p.parseType(type)) return mlir::failure();
        if (mlir::isa<VoidType>(type))
            // An explicit !p4hir.void means also no return type.
            optionalReturnType = {};
        else
            // Otherwise use the actual type.
            optionalReturnType = type;
        if (p.parseLParen()) return mlir::failure();
    }

    // `(` `)`
    if (succeeded(p.parseOptionalRParen())) return mlir::success();

    if (p.parseCommaSeparatedList([&]() -> ParseResult {
            mlir::Type type;
            if (p.parseType(type)) return mlir::failure();
            params.push_back(type);
            return mlir::success();
        }))
        return mlir::failure();

    return p.parseRParen();
}

static void printFuncType(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params,
                          mlir::Type optionalReturnType) {
    if (optionalReturnType) p << optionalReturnType << ' ';
    p << '(';
    llvm::interleaveComma(params, p, [&p](mlir::Type type) { p.printType(type); });
    p << ')';
}

// Return the actual return type or an explicit !p4hir.void if the function does
// not return anything
mlir::Type FuncType::getReturnType() const {
    if (isVoid()) return P4HIR::VoidType::get(getContext());
    return static_cast<detail::FuncTypeStorage *>(getImpl())->optionalReturnType;
}

/// Returns the result type of the function as an ArrayRef, enabling better
/// integration with generic MLIR utilities.
llvm::ArrayRef<mlir::Type> FuncType::getReturnTypes() const {
    if (isVoid()) return {};
    return static_cast<detail::FuncTypeStorage *>(getImpl())->optionalReturnType;
}

// Whether the function returns void
bool FuncType::isVoid() const {
    auto rt = static_cast<detail::FuncTypeStorage *>(getImpl())->optionalReturnType;
    assert(!rt || !mlir::isa<VoidType>(rt) &&
                      "The return type for a function returning void should be empty "
                      "instead of a real !p4hir.void");
    return !rt;
}

namespace P4::P4MLIR::P4HIR {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
    return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) { return llvm::hash_combine(fi.name, fi.type); }
}  // namespace P4::P4MLIR::P4HIR

/// Parse a list of unique field names and types within <> plus name. E.g.:
/// <name, foo: i7, bar: i8>
static ParseResult parseFields(AsmParser &p, std::string &name,
                               SmallVectorImpl<FieldInfo> &parameters) {
    llvm::StringSet<> nameSet;
    bool hasDuplicateName = false;
    bool parsedName = false;
    auto parseResult =
        p.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            // First, try to parse name
            if (!parsedName) {
                if (p.parseKeywordOrString(&name)) return failure();
                parsedName = true;
                return success();
            }

            // Parse fields
            std::string fieldName;
            Type fieldType;

            auto fieldLoc = p.getCurrentLocation();
            if (p.parseKeywordOrString(&fieldName) || p.parseColon() || p.parseType(fieldType))
                return failure();

            if (!nameSet.insert(fieldName).second) {
                p.emitError(fieldLoc, "duplicate field name \'" + name + "\'");
                // Continue parsing to print all duplicates, but make sure to error
                // eventually
                hasDuplicateName = true;
            }

            parameters.push_back(FieldInfo{StringAttr::get(p.getContext(), fieldName), fieldType});
            return success();
        });

    if (hasDuplicateName) return failure();
    return parseResult;
}

/// Print out a list of named fields surrounded by <>.
static void printFields(AsmPrinter &p, StringRef name, ArrayRef<FieldInfo> fields) {
    p << '<';
    p.printString(name);
    if (!fields.empty()) p << ", ";
    llvm::interleaveComma(fields, p, [&](const FieldInfo &field) {
        p.printKeywordOrString(field.name.getValue());
        p << ": " << field.type;
    });
    p << ">";
}

Type StructType::parse(AsmParser &p) {
    llvm::SmallVector<FieldInfo, 4> parameters;
    std::string name;
    if (parseFields(p, name, parameters)) return {};
    return get(p.getContext(), name, parameters);
}

Type HeaderType::parse(AsmParser &p) {
    llvm::SmallVector<FieldInfo, 4> parameters;
    std::string name;
    if (parseFields(p, name, parameters)) return {};
    return get(p.getContext(), name, parameters);
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                 ArrayRef<FieldInfo> elements) {
    llvm::SmallDenseSet<StringAttr> fieldNameSet;
    LogicalResult result = success();
    fieldNameSet.reserve(elements.size());
    for (const auto &elt : elements)
        if (!fieldNameSet.insert(elt.name).second) {
            result = failure();
            emitError() << "duplicate field name '" << elt.name.getValue()
                        << "' in p4hir.struct type";
        }
    return result;
}

LogicalResult HeaderType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                 ArrayRef<FieldInfo> elements) {
    llvm::SmallDenseSet<StringAttr> fieldNameSet;
    LogicalResult result = success();
    fieldNameSet.reserve(elements.size());
    for (const auto &elt : elements)
        if (!fieldNameSet.insert(elt.name).second) {
            result = failure();
            emitError() << "duplicate field name '" << elt.name.getValue()
                        << "' in p4hir.header type";
        }

    if (elements.empty()) emitError() << "empty p4hir.header type";

    if (elements.back().name != validityBit ||
        !mlir::isa<P4HIR::ValidBitType>(elements.back().type))
        emitError() << "last field of p4hir.header type should be validity bit";

    // TODO: Check field types & nesting

    return result;
}

void StructType::print(AsmPrinter &p) const { printFields(p, getName(), getElements()); }
void HeaderType::print(AsmPrinter &p) const { printFields(p, getName(), getElements()); }

HeaderType HeaderType::get(mlir::MLIRContext *context, llvm::StringRef name,
                           llvm::ArrayRef<FieldInfo> fields) {
    llvm::SmallVector<FieldInfo, 4> realFields(fields);
    realFields.push_back(
        {mlir::StringAttr::get(context, validityBit), P4HIR::ValidBitType::get(context)});

    return Base::get(context, name, realFields);
}

Type EnumType::parse(AsmParser &p) {
    std::string name;
    llvm::SmallVector<Attribute> fields;
    bool parsedName = false;
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
            // First, try to parse name
            if (!parsedName) {
                if (p.parseKeywordOrString(&name)) return failure();
                parsedName = true;
                return success();
            }

            StringRef caseName;
            if (p.parseKeyword(&caseName)) return failure();
            fields.push_back(StringAttr::get(p.getContext(), name));
            return success();
        }))
        return {};

    return get(p.getContext(), name, ArrayAttr::get(p.getContext(), fields));
}

void EnumType::print(AsmPrinter &p) const {
    auto fields = getFields();
    p << '<';
    p.printString(getName());
    if (!fields.empty()) p << ", ";
    llvm::interleaveComma(fields, p, [&](Attribute enumerator) {
        p << mlir::cast<StringAttr>(enumerator).getValue();
    });
    p << ">";
}

std::optional<size_t> EnumType::indexOf(mlir::StringRef field) {
    for (auto it : llvm::enumerate(getFields()))
        if (mlir::cast<StringAttr>(it.value()).getValue() == field) return it.index();
    return {};
}

Type ErrorType::parse(AsmParser &p) {
    llvm::SmallVector<Attribute> fields;
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
            StringRef caseName;
            if (p.parseKeyword(&caseName)) return failure();
            fields.push_back(StringAttr::get(p.getContext(), name));
            return success();
        }))
        return {};

    return get(p.getContext(), ArrayAttr::get(p.getContext(), fields));
}

void ErrorType::print(AsmPrinter &p) const {
    auto fields = getFields();
    p << '<';
    llvm::interleaveComma(fields, p, [&](Attribute enumerator) {
        p << mlir::cast<StringAttr>(enumerator).getValue();
    });
    p << ">";
}

std::optional<size_t> ErrorType::indexOf(mlir::StringRef field) {
    for (auto it : llvm::enumerate(getFields()))
        if (mlir::cast<StringAttr>(it.value()).getValue() == field) return it.index();
    return {};
}

void SerEnumType::print(AsmPrinter &p) const {
    auto fields = getFields();
    p << '<';
    p.printString(getName());
    p << ", ";
    p.printType(getType());
    if (!fields.empty()) p << ", ";
    llvm::interleaveComma(fields, p, [&](NamedAttribute enumerator) {
        p.printKeywordOrString(enumerator.getName());
        p << " : ";
        p.printAttribute(enumerator.getValue());
    });
    p << ">";
}

Type SerEnumType::parse(AsmParser &p) {
    std::string name;
    llvm::SmallVector<NamedAttribute> fields;
    P4HIR::BitsType type;

    // Parse "<name, type, " part
    if (p.parseLess() || p.parseKeywordOrString(&name) || p.parseComma() ||
        p.parseCustomTypeWithFallback<P4HIR::BitsType>(type) || p.parseComma())
        return {};

    if (p.parseCommaSeparatedList([&]() {
            StringRef caseName;
            P4HIR::IntAttr attr;
            // Parse fields "name : #value"
            if (p.parseKeyword(&caseName) || p.parseColon() ||
                p.parseCustomAttributeWithFallback<P4HIR::IntAttr>(attr))
                return failure();

            fields.emplace_back(StringAttr::get(p.getContext(), caseName), attr);
            return success();
        }))
        return {};

    // Parse closing >
    if (p.parseGreater()) return {};

    return get(name, type, fields);
}

Type ValidBitType::parse(mlir::AsmParser &parser) { return get(parser.getContext()); }

void ValidBitType::print(mlir::AsmPrinter &printer) const {}

void P4HIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"  // NOLINT
        >();
}
