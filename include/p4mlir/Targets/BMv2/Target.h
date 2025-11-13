#include "llvm/Support/JSON.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace P4::P4MLIR {

mlir::LogicalResult bmv2irToJson(mlir::ModuleOp moduleOp, mlir::raw_ostream &output);

// Serializes a BMv2IR module to its final JSON representation
mlir::FailureOr<llvm::json::Value> bmv2irToJson(mlir::ModuleOp moduleOp);

void registerToBMv2JSONTranslation();

}  // namespace P4::P4MLIR
