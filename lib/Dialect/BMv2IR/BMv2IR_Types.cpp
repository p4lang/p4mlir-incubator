#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

template <>
struct mlir::FieldParser<P4::P4MLIR::BMv2IR::FieldInfo> {
    static FailureOr<P4::P4MLIR::BMv2IR::FieldInfo> parse(AsmParser &parser) {
        StringRef name;

        if (failed(parser.parseKeyword(&name))) return failure();
        if (parser.parseColon()) return failure();
        Type ty;
        if (parser.parseType(ty)) return failure();

        return BMv2IR::FieldInfo(StringAttr::get(parser.getContext(), name), ty);
    }
};

constexpr unsigned bitsInByte = 8;
static unsigned computeTotalHeaderLenghtInBits(ArrayRef<BMv2IR::FieldInfo> fields) {
    unsigned total = 0;
    for (const auto &field : fields) {
        total += TypeSwitch<Type, unsigned>(field.type)
                     .Case<P4HIR::BitsType>([](P4HIR::BitsType bitTy) { return bitTy.getWidth(); })
                     .Case<P4HIR::VarBitsType>(
                         [](P4HIR::VarBitsType varBitTy) { return varBitTy.getMaxWidth(); })
                     .Default([](auto) -> unsigned {
                         llvm_unreachable("Unsupported field in BMv2 header");
                     });
    }
    return total;
}

bool BMv2IR::HeaderType::isAllowedFieldType(Type ty) {
    return isa<P4HIR::BitsType, P4HIR::VarBitsType>(ty);
}

llvm::LogicalResult BMv2IR::HeaderType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::StringRef name,
    ::llvm::ArrayRef<BMv2IR::FieldInfo> fields, unsigned max_length) {
    if (llvm::any_of(fields, [](BMv2IR::FieldInfo field) {
            return !BMv2IR::HeaderType::isAllowedFieldType(field.type);
        })) {
        emitError() << "Only bits and varbits are allowed in BMv2 headers";
        return failure();
    }

    unsigned lenInBits = computeTotalHeaderLenghtInBits(fields);
    if (lenInBits % bitsInByte != 0) {
        emitError() << "Expected total size of a header to be byte-sized";
        return failure();
    }

    unsigned numVarBits = llvm::count_if(
        fields, [](BMv2IR::FieldInfo field) { return isa<P4HIR::VarBitsType>(field.type); });
    if (numVarBits > 1) {
        emitError() << "Expected at most one field with dynamic size.";
        return failure();
    }

    auto computedMaxLength = computeMaxLength(fields);
    if (max_length != computedMaxLength) {
        emitError() << "Max length mismatch: expected " << computedMaxLength << " got "
                    << max_length;
        return failure();
    }

    return success();
}

bool BMv2IR::HeaderType::hasField(StringRef name) {
    bool hasField =
        llvm::any_of(getFields(), [&](BMv2IR::FieldInfo info) { return info.name == name; });
    return hasField || name == validBitFieldName;
}

FailureOr<BMv2IR::FieldInfo> BMv2IR::HeaderType::getField(StringRef name) {
    for (auto field : getFields()) {
        if (field.name == name) return field;
    }
    return failure();
}

unsigned BMv2IR::HeaderType::computeMaxLength(ArrayRef<BMv2IR::FieldInfo> fields) {
    unsigned lenInBits = computeTotalHeaderLenghtInBits(fields);
    return lenInBits / bitsInByte;
}

llvm::LogicalResult BMv2IR::HeaderUnionType::verify(llvm::function_ref<mlir::InFlightDiagnostic()>,
                                                    llvm::StringRef,
                                                    llvm::ArrayRef<P4::P4MLIR::BMv2IR::FieldInfo>) {
    // TODO: check that every field is of header type
    return success();
}
#define GET_TYPEDEF_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.cpp.inc"

llvm::hash_code P4::P4MLIR::BMv2IR::hash_value(P4::P4MLIR::BMv2IR::FieldInfo f) {
    return llvm::hash_value(f.name.getValue());
}

void BMv2IR::BMv2IRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.cpp.inc"  // NOLINT
        >();
}
