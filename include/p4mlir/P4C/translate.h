#ifndef INCLUDE_P4MLIR_P4C_TRANSLATE_H_
#define INCLUDE_P4MLIR_P4C_TRANSLATE_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

namespace P4 {
namespace IR {
class P4Program;
}  // namespace IR
class TypeMap;
}  // namespace P4

namespace P4::P4MLIR {
mlir::OwningOpRef<mlir::ModuleOp> toMLIR(mlir::MLIRContext &context,
                                         const P4::IR::P4Program *program, P4::TypeMap *typeMap);
}  // namespace P4::P4MLIR

#endif  // INCLUDE_P4MLIR_P4C_TRANSLATE_H_
