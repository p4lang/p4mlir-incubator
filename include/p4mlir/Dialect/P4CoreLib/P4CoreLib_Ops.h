#ifndef P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H
#define P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace mlir::P4MLIR::P4CoreLib {

struct PacketResource : public mlir::SideEffects::Resource::Base<PacketResource> {
    mlir::StringRef getName() final { return "Packet"; }
};

}  // namespace mlir::P4MLIR::P4CoreLib

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h.inc"

#endif  // P4MLIR_DIALECT_P4CORELIB_P4CORELIB_OPS_H
