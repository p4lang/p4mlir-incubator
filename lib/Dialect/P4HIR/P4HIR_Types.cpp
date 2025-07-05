#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;
using namespace P4::P4MLIR::P4HIR::detail;

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params,
                                       mlir::Type &optionalResultType);
static mlir::ParseResult parseFuncType(mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params);

static mlir::ParseResult parseCtorType(
    mlir::AsmParser &p, llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>> &params,
    mlir::Type &resultType);

static mlir::ParseResult parseFieldInfo(mlir::AsmParser &p, FailureOr<FieldInfo> &fi);

static mlir::ParseResult parseArray(mlir::AsmParser &p, size_t &size, mlir::Type &elementType);

static void printFuncType(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params,
                          mlir::Type optionalResultType = {});
static void printCtorType(mlir::AsmPrinter &p,
                          mlir::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> params,
                          mlir::Type resultType);
static void printFieldInfo(mlir::AsmPrinter &p, const FieldInfo &fi);
static void printArray(mlir::AsmPrinter &p, size_t size, mlir::Type elementType);
static void printArray(mlir::AsmPrinter &p, ArrayType arrayType);

namespace P4::P4MLIR::P4HIR {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
    return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) {
    return llvm::hash_combine(fi.name, fi.type, fi.annotations);
}

mlir::TypedAttr getStructLikeDefaultValue(StructLikeTypeInterface type) {
    llvm::SmallVector<mlir::Attribute> fieldDefaults;
    for (auto field : type.getFields()) {
        auto hasDef = mlir::dyn_cast<HasDefaultValue>(field.type);
        if (!hasDef) return nullptr;
        auto defValue = hasDef.getDefaultValue();
        if (!defValue) return nullptr;
        fieldDefaults.push_back(defValue);
    }

    return AggAttr::get(type, mlir::ArrayAttr::get(type.getContext(), fieldDefaults));
}
}  // namespace P4::P4MLIR::P4HIR

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

mlir::TypedAttr BitsType::getDefaultValue() {
    return P4HIR::IntAttr::get(*this, APInt::getZero(getWidth()));
}

mlir::TypedAttr InfIntType::getDefaultValue() { return P4HIR::IntAttr::get(*this, 0); }

mlir::TypedAttr VarBitsType::getDefaultValue() {
    return P4HIR::IntAttr::get(*this, APInt::getZero(getMaxWidth()));
}

Type BoolType::parse(mlir::AsmParser &parser) { return get(parser.getContext()); }

void BoolType::print(mlir::AsmPrinter &printer) const {}

mlir::TypedAttr BoolType::getDefaultValue() { return P4HIR::BoolAttr::get(getContext(), false); }

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

static mlir::ParseResult parseCtorType(
    mlir::AsmParser &p, llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>> &params,
    mlir::Type &returnType) {
    if (p.parseType(returnType) || p.parseLParen()) return mlir::failure();

    // `(` `)`
    if (succeeded(p.parseOptionalRParen())) return mlir::success();

    if (p.parseCommaSeparatedList([&]() -> ParseResult {
            std::string name;
            mlir::Type type;
            if (p.parseKeywordOrString(&name) || p.parseColon() || p.parseType(type))
                return mlir::failure();
            params.emplace_back(mlir::StringAttr::get(p.getContext(), name), type);
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

static void printCtorType(mlir::AsmPrinter &p,
                          mlir::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> params,
                          mlir::Type returnType) {
    p << returnType << ' ';
    p << '(';
    llvm::interleaveComma(params, p, [&p](std::pair<mlir::StringAttr, mlir::Type> namedType) {
        p << namedType.first << " : ";
        p.printType(namedType.second);
    });
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

static ParseResult parseFieldInfo(AsmParser &p, FailureOr<FieldInfo> &field) {
    // Parse fields
    std::string fieldName;
    Type fieldType;
    mlir::NamedAttrList fieldAnnotations;

    if (p.parseKeywordOrString(&fieldName) || p.parseColon() || p.parseType(fieldType) ||
        p.parseOptionalAttrDict(fieldAnnotations))
        return failure();

    field.emplace(StringAttr::get(p.getContext(), fieldName), fieldType,
                  fieldAnnotations.getDictionary(p.getContext()));

    return success();
}

/// Parse a list of unique field names and types within <> plus name. E.g.:
/// <name, foo: i7, bar: i8>
static ParseResult parseFields(AsmParser &p, std::string &name,
                               SmallVectorImpl<FieldInfo> &parameters,
                               mlir::DictionaryAttr &annotations) {
    llvm::StringSet<> nameSet;
    mlir::NamedAttrList annList;
    bool hasDuplicateName = false;
    bool parsedName = false;
    auto parseResult =
        p.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            // First, try to parse name
            if (!parsedName) {
                if (p.parseKeywordOrString(&name) || p.parseOptionalAttrDict(annList))
                    return failure();
                parsedName = true;
                annotations = annList.getDictionary(p.getContext());
                return success();
            }

            // Parse fields
            FailureOr<FieldInfo> field;
            auto fieldLoc = p.getCurrentLocation();
            if (parseFieldInfo(p, field)) return failure();

            if (!nameSet.insert(field->name).second) {
                p.emitError(fieldLoc, "duplicate field name \'" + name + "\'");
                // Continue parsing to print all duplicates, but make sure to error
                // eventually
                hasDuplicateName = true;
            }

            parameters.emplace_back(*field);
            return success();
        });

    if (hasDuplicateName) return failure();
    return parseResult;
}

static void printFieldInfo(AsmPrinter &p, const FieldInfo &field) {
    p.printKeywordOrString(field.name.getValue());
    p << ": " << field.type;
    if (field.annotations && !field.annotations.empty()) {
        p << " ";
        p.printAttributeWithoutType(field.annotations);
    }
}

/// Print out a list of named fields surrounded by <>.
static void printFields(AsmPrinter &p, StringRef name, ArrayRef<FieldInfo> fields,
                        mlir::DictionaryAttr annotations) {
    p << '<';
    p.printString(name);
    if (annotations && !annotations.empty()) {
        p << ' ';
        p.printAttributeWithoutType(annotations);
    }
    if (!fields.empty()) p << ", ";
    llvm::interleaveComma(fields, p, [&](const FieldInfo &field) { printFieldInfo(p, field); });
    p << ">";
}

Type StructType::parse(AsmParser &p) {
    llvm::SmallVector<FieldInfo, 4> parameters;
    std::string name;
    mlir::DictionaryAttr annotations;
    if (parseFields(p, name, parameters, annotations)) return {};
    return get(p.getContext(), name, parameters, annotations);
}

mlir::TypedAttr ValidBitType::getDefaultValue() {
    return P4HIR::ValidityBitAttr::get(getContext(), ValidityBit::Invalid);
};

Type HeaderType::parse(AsmParser &p) {
    llvm::SmallVector<FieldInfo, 4> parameters;
    std::string name;
    mlir::DictionaryAttr annotations;
    if (parseFields(p, name, parameters, annotations)) return {};
    // Do not use our own get() here as it adds __validity bit. And we do have it already.
    return Base::get(p.getContext(), name, parameters,
                     annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
}

Type HeaderUnionType::parse(AsmParser &p) {
    llvm::SmallVector<FieldInfo, 4> parameters;
    std::string name;
    mlir::DictionaryAttr annotations;
    if (parseFields(p, name, parameters, annotations)) return {};
    return get(p.getContext(), name, parameters, mlir::DictionaryAttr());
}

Type HeaderStackType::parse(AsmParser &p) {
    size_t sz;
    mlir::Type elementType;

    if (p.parseLess() || parseArray(p, sz, elementType) || p.parseGreater()) return nullptr;

    if (!mlir::isa<P4HIR::HeaderType, P4HIR::HeaderUnionType>(elementType)) {
        p.emitError(p.getCurrentLocation(), "header stack should be made out of headers");
        return nullptr;
    }

    return get(p.getContext(), sz, mlir::cast<P4HIR::StructLikeTypeInterface>(elementType));
}

void HeaderStackType::print(AsmPrinter &p) const {
    p << '<';
    FieldInfo dataField = getElements().front();
    printArray(p, mlir::cast<P4HIR::ArrayType>(dataField.type));
    p << '>';
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                 ArrayRef<FieldInfo> elements, DictionaryAttr) {
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
                                 ArrayRef<FieldInfo> elements, DictionaryAttr) {
    if (elements.empty()) {
        emitError() << "empty p4hir.header type";
        return failure();
    }

    LogicalResult result = success();
    llvm::SmallDenseSet<StringAttr> fieldNameSet;
    fieldNameSet.reserve(elements.size());
    for (const auto &elt : elements) {
        if (!fieldNameSet.insert(elt.name).second) {
            result = failure();
            emitError() << "duplicate field name '" << elt.name.getValue()
                        << "' in p4hir.header type";
        }
    }

    auto lastField = elements.back();
    if (lastField.name != validityBit || !mlir::isa<P4HIR::ValidBitType>(lastField.type)) {
        result = failure();
        emitError() << "the last field of p4hir.header type should be validity bit, but got"
                    << lastField.name << " of type " << lastField.type;
    }

    auto varbitCount = llvm::count_if(
        elements, [](const FieldInfo &field) { return mlir::isa<P4HIR::VarBitsType>(field.type); });

    if (varbitCount > 1) {
        result = failure();
        emitError() << "only one varbit field is allowed in p4hir.header type";
    }

    // If a varbit field is found, ensure it is the last “data” field
    // immediately before the validity field
    if (varbitCount > 0) {
        // Find the index of the varbit field
        auto varbitField = llvm::find_if(elements, [](const FieldInfo &field) {
            return mlir::isa<P4HIR::VarBitsType>(field.type);
        });

        size_t varbitIndex = std::distance(elements.begin(), varbitField);
        size_t expectedIndex = elements.size() - 2;  // The index is right before validity bit

        if (varbitIndex != expectedIndex) {
            result = failure();
            emitError() << "varbit field " << varbitField->name
                        << " must be immediately before the validity field in p4hir.header type";
        }
    }

    // Helper function to verify inner types recursively, inspecting nested structs
    std::function<void(ArrayRef<FieldInfo>, bool)> verifyInnerTypes =
        [&](ArrayRef<FieldInfo> elements, bool isHeader) -> void {
        for (const auto &elt : elements) {
            auto fieldType = elt.type;
            if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(fieldType))
                fieldType = aliasType.getCanonicalType();

            // varbit and validity bit are allowed at the top (header) level only
            if (isHeader && mlir::isa<P4HIR::VarBitsType, P4HIR::ValidBitType>(fieldType)) {
                continue;
            }
            if (mlir::isa<P4HIR::BitsType, P4HIR::BoolType, P4HIR::SerEnumType>(fieldType)) {
                continue;
            }
            if (const auto structType = mlir::dyn_cast<P4HIR::StructType>(fieldType)) {
                verifyInnerTypes(structType.getElements(), false);
                continue;
            }

            result = failure();
            emitError() << "field name " << elt.name << " is of type '" << fieldType
                        << "' that is not allowed in p4hir.header type";
        }
    };

    verifyInnerTypes(elements, true);

    return result;
}

LogicalResult HeaderUnionType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                      ArrayRef<FieldInfo> elements, DictionaryAttr) {
    llvm::SmallDenseSet<StringAttr> fieldNameSet;
    LogicalResult result = success();
    fieldNameSet.reserve(elements.size());
    for (const auto &elt : elements) {
        if (!fieldNameSet.insert(elt.name).second) {
            result = failure();
            emitError() << "duplicate field name '" << elt.name.getValue()
                        << "' in p4hir.header_union type";
        }

        // Check that all elements are header types
        if (!mlir::isa<HeaderType>(elt.type)) {
            result = failure();
            emitError() << "header_union field '" << elt.name.getValue()
                        << "' must be a header type";
        }
    }

    if (elements.empty()) {
        emitError() << "empty p4hir.header_union type";
        return failure();
    }

    return result;
}

LogicalResult HeaderStackType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                      ArrayRef<FieldInfo> elements, DictionaryAttr) {
    if (elements.size() != 2) {
        emitError() << "invalid struct size for header stack";
        return failure();
    }

    FieldInfo dataField = elements.front();
    FieldInfo nextIndexField = elements.back();

    auto arrayType = mlir::dyn_cast<P4HIR::ArrayType>(dataField.type);
    if (!arrayType ||
        !mlir::isa<P4HIR::HeaderType, P4HIR::HeaderUnionType>(arrayType.getElementType())) {
        emitError() << "header stack data must be an array of headers";
        return failure();
    }

    // Basically, check name and type
    if (nextIndexField.name != nextIndexFieldName) {
        emitError() << "header stack nextIndex field should be named " << nextIndexFieldName
                    << ", got " << nextIndexField.name;
        return failure();
    }

    if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(nextIndexField.type)) {
        if (bitsType.getIsSigned() || bitsType.getWidth() != 32) {
            emitError() << "header stack nextIndex field must be of bit<32> type";
            return failure();
        }
    }

    return success();
}

HeaderStackType HeaderStackType::get(mlir::MLIRContext *context, size_t sz,
                                     P4HIR::StructLikeTypeInterface elementType) {
    assert((mlir::isa<P4HIR::HeaderType, P4HIR::HeaderUnionType>(elementType)) &&
           "expected header or header union type for stack element");
    auto dataType = P4HIR::ArrayType::get(context, sz, elementType);
    FieldInfo dataField(mlir::StringAttr::get(context, dataFieldName), dataType);
    FieldInfo nextIndexField(mlir::StringAttr::get(context, nextIndexFieldName),
                             P4HIR::BitsType::get(context, 32, false));

    return Base::get(context, "hs", ArrayRef{dataField, nextIndexField}, nullptr);
}

ArrayType HeaderStackType::getDataType() {
    return mlir::cast<ArrayType>(getElements().front().type);
}
size_t HeaderStackType::getArraySize() { return getDataType().getSize(); }
StructLikeTypeInterface HeaderStackType::getArrayElementType() {
    return mlir::cast<StructLikeTypeInterface>(getDataType().getElementType());
}

void StructType::print(AsmPrinter &p) const {
    printFields(p, getName(), getElements(), getAnnotations());
}
void HeaderType::print(AsmPrinter &p) const {
    printFields(p, getName(), getElements(), getAnnotations());
}
void HeaderUnionType::print(AsmPrinter &p) const {
    printFields(p, getName(), getElements(), getAnnotations());
}

HeaderType HeaderType::get(mlir::MLIRContext *context, llvm::StringRef name,
                           llvm::ArrayRef<FieldInfo> fields, mlir::DictionaryAttr annotations) {
    llvm::SmallVector<FieldInfo, 4> realFields(fields);
    realFields.emplace_back(mlir::StringAttr::get(context, validityBit),
                            P4HIR::ValidBitType::get(context));

    return Base::get(context, name, realFields,
                     annotations && !annotations.empty() ? annotations : mlir::DictionaryAttr());
}

Type EnumType::parse(AsmParser &p) {
    std::string name;
    llvm::SmallVector<Attribute> fields;
    bool parsedName = false;
    mlir::NamedAttrList annotations;

    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
            // First, try to parse name
            if (!parsedName) {
                bool hasName = llvm::succeeded(p.parseOptionalString(&name));

                if (p.parseOptionalAttrDict(annotations)) return failure();

                parsedName = true;
                // If there is name, then proceed to the next field
                if (hasName || !annotations.empty()) return success();
            }

            StringRef caseName;
            if (p.parseKeyword(&caseName)) return failure();
            fields.push_back(StringAttr::get(p.getContext(), caseName));
            return success();
        }))
        return {};

    return get(p.getContext(), name, ArrayAttr::get(p.getContext(), fields),
               annotations.getDictionary(p.getContext()));
}

void EnumType::print(AsmPrinter &p) const {
    auto fields = getFields();
    p << '<';
    bool firstField = true;
    if (!getName().empty()) {
        p.printString(getName());
        firstField = false;
    }
    if (auto annotations = getAnnotations(); annotations && !annotations.empty()) {
        if (!firstField) p << ' ';
        p.printAttributeWithoutType(annotations);
        firstField = false;
    }
    if (!fields.empty() && !firstField) p << ", ";
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

mlir::TypedAttr EnumType::getDefaultValue() {
    auto fields = getFields();
    if (fields.empty()) return nullptr;

    return P4HIR::EnumFieldAttr::get(*this, mlir::cast<mlir::StringAttr>(fields[0]));
}

Type ErrorType::parse(AsmParser &p) {
    llvm::SmallVector<Attribute> fields;
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
            StringRef caseName;
            if (p.parseKeyword(&caseName)) return failure();
            fields.push_back(StringAttr::get(p.getContext(), caseName));
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
    if (auto annotations = getAnnotations(); annotations && !annotations.empty()) {
        p << ' ';
        p.printAttributeWithoutType(annotations);
    }
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
    mlir::NamedAttrList annotations;

    // Parse "<name, type, " part
    if (p.parseLess() || p.parseKeywordOrString(&name) || p.parseOptionalAttrDict(annotations) ||
        p.parseComma() || p.parseCustomTypeWithFallback<P4HIR::BitsType>(type) || p.parseComma())
        return {};

    // Parse comma separated set of fields "name : #value"
    if (p.parseCommaSeparatedList([&]() {
            StringRef caseName;
            P4HIR::IntAttr attr;
            if (p.parseKeyword(&caseName) || p.parseColon() ||
                p.parseCustomAttributeWithFallback<P4HIR::IntAttr>(attr))
                return failure();

            fields.emplace_back(StringAttr::get(p.getContext(), caseName), attr);
            return success();
        }))
        return {};

    // Parse closing >
    if (p.parseGreater()) return {};

    return get(name, type, fields, annotations.getDictionary(p.getContext()));
}

// The spec says that the default value is zero even if there is no zero field.
// This looks like a terrible idea as this is non-representable w/o casting to
// underlying type. We return null instead here.
mlir::TypedAttr SerEnumType::getDefaultValue() {
    for (auto field : getFields()) {
        auto val = mlir::cast<P4HIR::IntAttr>(field.getValue());
        if (val.getValue().isZero()) return P4HIR::EnumFieldAttr::get(*this, field.getName());
    }

    return nullptr;
}

Type ValidBitType::parse(mlir::AsmParser &parser) { return get(parser.getContext()); }

void ValidBitType::print(mlir::AsmPrinter &printer) const {}

static mlir::ParseResult parseArray(mlir::AsmParser &p, size_t &size, mlir::Type &elementType) {
    if (p.parseInteger(size) || p.parseXInDimensionList() || p.parseType(elementType))
        return failure();

    return success();
}

void printArray(mlir::AsmPrinter &p, size_t size, mlir::Type elementType) {
    p << size << 'x';
    p.printType(elementType);
}

void printArray(mlir::AsmPrinter &p, ArrayType arrayType) {
    printArray(p, arrayType.getSize(), arrayType.getElementType());
}

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError, size_t size,
                                Type elementType) {
    if (mlir::isa<ReferenceType>(elementType))
        return emitError() << "p4hir.array cannot contain reference types";
    return success();
}

unsigned ArrayType::getMaxFieldID() const {
    return getSize() * (FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<Type, unsigned> ArrayType::getSubTypeByFieldID(unsigned fieldID) const {
    if (fieldID == 0) return {*this, 0};
    return {getElementType(), getIndexAndSubfieldID(fieldID).second};
}

std::pair<unsigned, bool> ArrayType::projectToChildFieldID(unsigned fieldID, unsigned index) const {
    auto childRoot = getFieldID(index);
    auto rangeEnd = index >= getSize() ? getMaxFieldID() : (getFieldID(index + 1) - 1);
    return std::make_pair(fieldID - childRoot, fieldID >= childRoot && fieldID <= rangeEnd);
}

unsigned ArrayType::getIndexForFieldID(unsigned fieldID) const {
    assert(fieldID && "fieldID must be at least 1");
    // Divide the field ID by the number of fieldID's per element.
    return (fieldID - 1) / (FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<unsigned, unsigned> ArrayType::getIndexAndSubfieldID(unsigned fieldID) const {
    auto index = getIndexForFieldID(fieldID);
    auto elementFieldID = getFieldID(index);
    return {index, fieldID - elementFieldID};
}

unsigned ArrayType::getFieldID(unsigned index) const {
    return 1 + index * (FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::optional<DenseMap<Attribute, Type>> ArrayType::getSubelementIndexMap() const {
    DenseMap<Attribute, Type> destructured;
    for (unsigned i = 0; i < getSize(); ++i)
        destructured.insert({IntegerAttr::get(IndexType::get(getContext()), i), getElementType()});
    return destructured;
}

Type ArrayType::getTypeAtIndex(Attribute) const { return getElementType(); }

mlir::TypedAttr ArrayType::getDefaultValue() {
    auto hasDef = mlir::dyn_cast<HasDefaultValue>(getElementType());
    if (!hasDef) return nullptr;
    auto defValue = hasDef.getDefaultValue();
    if (!defValue) return nullptr;

    llvm::SmallVector<Attribute> defVals(getSize(), defValue);
    return P4HIR::AggAttr::get(*this, mlir::ArrayAttr::get(getContext(), defVals));
}

void P4HIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"  // NOLINT
        >();
}
