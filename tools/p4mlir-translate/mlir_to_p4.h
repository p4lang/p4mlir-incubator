#ifndef TOOLS_P4MLIR_TRANSLATE_MLIR_TO_P4_H_
#define MLIR_TO_P4_H

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

namespace P4::P4MLIR {

class MlirToP4 {
 public:
    MlirToP4(mlir::ModuleOp module);
    std::string convert();

 private:
    mlir::ModuleOp module;
    // Additional helper methods for specific MLIR operations
};

}  // namespace P4::P4MLIR

#endif  // TOOLS_P4MLIR_TRANSLATE_MLIR_TO_P4_H_
