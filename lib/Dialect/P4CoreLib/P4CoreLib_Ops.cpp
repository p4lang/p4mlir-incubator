#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"

#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"

using namespace mlir;
using namespace P4::P4MLIR;

void P4CoreLib::PacketLengthOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "packet_len");
}

void P4CoreLib::PacketLookAheadOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "lookahead");
}

OpFoldResult P4CoreLib::StaticAssertOp::fold(FoldAdaptor adaptor) {
    // This fold handles the true case early during canonicalization, without
    // needing the full EvaluateStaticAssertPass pipeline. False assertions are
    // intentionally left unhandled here so the pass can emit proper error messages.
    auto condAttr = adaptor.getCond();
    if (!condAttr)
        return {};  // Condition not constant yet

    auto boolAttr = llvm::dyn_cast<P4HIR::BoolAttr>(condAttr);
    if (!boolAttr)
        return {};  // Not a boolean

    if (boolAttr.getValue()) {
        // Assertion passed -> fold to constant true
        return P4HIR::BoolAttr::get(getContext(), true);
    }

    // Assertion failed -> seperate pass will handle the error
    return {};
}

void P4CoreLib::P4CoreLibDialect::initialize() {
    registerTypes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.cpp.inc"  // NOLINT
        >();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.cpp.inc"  // NOLINT
