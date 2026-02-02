#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"

#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.cpp.inc"

using namespace mlir;
using namespace P4::P4MLIR;
using namespace P4::P4MLIR::BMv2IR;

LogicalResult MatchKeyAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                   TableMatchKind kind, mlir::Attribute first, Attribute second) {
    switch (kind) {
        case TableMatchKind::Exact: {
            if (!isa<P4HIR::IntAttr>(first))
                return emitError() << "Expected IntAttr for Exact kind";
            return success();
        }
        case TableMatchKind::Ternary:
        case TableMatchKind::Range: {
            if (!isa<P4HIR::IntAttr>(first) || !isa<P4HIR::IntAttr>(second))
                return emitError() << "Expected IntAttr for Range kind";
            return success();
        }
        case TableMatchKind::LPM: {
            if (!isa<P4HIR::IntAttr>(first)) return emitError() << "Expected IntAttr for LPM kind";
            if (!isa<IntegerAttr>(second))
                return emitError() << "Expected IntegerAttr for LPM kind";
            return success();
        }
        default: {
            return emitError() << "Unsupported TableMatchKind";
        }
    }
    return success();
}

#include "p4mlir/Dialect/BMv2IR/BMv2IR_EnumAttrs.cpp.inc"
void BMv2IRDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.cpp.inc"  // NOLINT
        >();
}
