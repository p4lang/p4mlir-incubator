#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
    if (mlir::isa<P4HIR::BoolAttr>(attrType)) {
        if (!mlir::isa<P4HIR::BoolType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.bool' for '" << attrType << "'";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr>(attrType)) {
        if (!mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") does not match value type (" << attrType << ")";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr, P4HIR::BoolAttr>(attrType)) return success();

    assert(isa<TypedAttr>(attrType) && "expected typed attribute");
    return op->emitOpError("constant with type ")
           << cast<TypedAttr>(attrType).getType() << " not supported";
}

LogicalResult P4HIR::ConstOp::verify() {
    // ODS already generates checks to make sure the result type is valid. We just
    // need to additionally check that the value's attribute type is consistent
    // with the result type.
    return checkConstantTypes(getOperation(), getType(), getValue());
}

LogicalResult P4HIR::UnaryOp::verify() {
    switch (getKind()) {
        case P4HIR::UnaryOpKind::Neg:
        case P4HIR::UnaryOpKind::UPlus:
        case P4HIR::UnaryOpKind::Cmpl:
        case P4HIR::UnaryOpKind::LNot:
            // Nothing to verify.
            return success();
    }

    llvm_unreachable("Unknown UnaryOp kind?");
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void P4HIR::AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                            ::mlir::Type ref, ::mlir::Type objectType, ::llvm::StringRef name) {
    odsState.addAttribute(getObjectTypeAttrName(odsState.name), ::mlir::TypeAttr::get(objectType));
    odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
    odsState.addTypes(ref);
}

void P4HIR::P4HIRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
        >();
}
