#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;
using namespace P4::P4MLIR::P4HIR::detail;

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, mlir::Type &optionalResultType,
                                       llvm::SmallVector<mlir::Type> &params);

static void printFuncType(mlir::AsmPrinter &p, mlir::Type optionalResultType,
                          mlir::ArrayRef<mlir::Type> params);

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

static mlir::ParseResult parseFuncType(mlir::AsmParser &p, mlir::Type &optionalReturnType,
                                       llvm::SmallVector<mlir::Type> &params) {
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

static void printFuncType(mlir::AsmPrinter &p, mlir::Type optionalReturnType,
                          mlir::ArrayRef<mlir::Type> params) {
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

namespace P4::P4MLIR::P4HIR::detail {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
    return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) { return llvm::hash_combine(fi.name, fi.type); }
}  // namespace P4::P4MLIR::P4HIR::detail

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

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef,
                                 ArrayRef<StructType::FieldInfo> elements) {
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

void StructType::print(AsmPrinter &p) const { printFields(p, getName(), getElements()); }

Type StructType::getFieldType(mlir::StringRef fieldName) {
    for (const auto &field : getElements())
        if (field.name == fieldName) return field.type;
    return Type();
}

std::optional<unsigned> StructType::getFieldIndex(mlir::StringRef fieldName) {
    ArrayRef<StructType::FieldInfo> elems = getElements();
    for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
        if (elems[idx].name == fieldName) return idx;
    return {};
}

std::optional<unsigned> StructType::getFieldIndex(mlir::StringAttr fieldName) {
    ArrayRef<StructType::FieldInfo> elems = getElements();
    for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
        if (elems[idx].name == fieldName) return idx;
    return {};
}

static std::pair<unsigned, SmallVector<unsigned>> getFieldIDsStruct(const StructType &st) {
    unsigned fieldID = 0;
    auto elements = st.getElements();
    SmallVector<unsigned> fieldIDs;
    fieldIDs.reserve(elements.size());
    for (auto &element : elements) {
        auto type = element.type;
        fieldID += 1;
        fieldIDs.push_back(fieldID);
        // Increment the field ID for the next field by the number of subfields.
        fieldID += FieldIdImpl::getMaxFieldID(type);
    }
    return {fieldID, fieldIDs};
}

std::pair<Type, unsigned> StructType::getSubTypeByFieldID(unsigned fieldID) const {
    if (fieldID == 0) return {*this, 0};
    auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
    auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
    auto subfieldIndex = std::distance(fieldIDs.begin(), it);
    auto subfieldType = getElements()[subfieldIndex].type;
    auto subfieldID = fieldID - fieldIDs[subfieldIndex];
    return {subfieldType, subfieldID};
}

Type StructType::getTypeAtIndex(Attribute index) const {
    auto indexAttr = llvm::dyn_cast<IntegerAttr>(index);
    if (!indexAttr) return {};

    return getSubTypeByFieldID(indexAttr.getInt()).first;
}

unsigned StructType::getFieldID(unsigned index) const {
    auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
    return fieldIDs[index];
}

unsigned StructType::getMaxFieldID() const {
    unsigned fieldID = 0;
    for (const auto &field : getElements()) fieldID += 1 + FieldIdImpl::getMaxFieldID(field.type);
    return fieldID;
}

unsigned StructType::getIndexForFieldID(unsigned fieldID) const {
    assert(!getElements().empty() && "struct must have >0 fields");
    auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
    auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
    return std::distance(fieldIDs.begin(), it);
}

std::pair<unsigned, unsigned> StructType::getIndexAndSubfieldID(unsigned fieldID) const {
    auto index = getIndexForFieldID(fieldID);
    auto elementFieldID = getFieldID(index);
    return {index, fieldID - elementFieldID};
}

std::optional<DenseMap<Attribute, Type>> StructType::getSubelementIndexMap() const {
    DenseMap<Attribute, Type> destructured;
    for (auto [i, field] : llvm::enumerate(getElements()))
        destructured.try_emplace(IntegerAttr::get(IndexType::get(getContext()), i), field.type);
    return destructured;
}

std::pair<unsigned, bool> StructType::projectToChildFieldID(unsigned fieldID,
                                                            unsigned index) const {
    auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
    auto childRoot = fieldIDs[index];
    auto rangeEnd = index + 1 >= getElements().size() ? maxId : (fieldIDs[index + 1] - 1);
    return std::make_pair(fieldID - childRoot, fieldID >= childRoot && fieldID <= rangeEnd);
}

void StructType::getInnerTypes(SmallVectorImpl<Type> &types) {
    for (const auto &field : getElements()) types.push_back(field.type);
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

void P4HIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.cpp.inc"  // NOLINT
        >();
}
