#ifndef LIB_UTILITIES_MLIR_TO_P4_H_
#define LIB_UTILITIES_MLIR_TO_P4_H_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

namespace P4::P4MLIR::Utilities {

struct MlirToP4ProgramOptions {
    /// The default indentation unit to use for the generated program (tabs vs 2,3,4 spaces..,).
    std::string indent_unit;

    MlirToP4ProgramOptions() : indent_unit("    ") {}
};

std::string convertMlirToP4(mlir::ModuleOp module, MlirToP4ProgramOptions options);

void convertMlirToP4(mlir::ModuleOp module, std::ostream &os, MlirToP4ProgramOptions options);

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_MLIR_TO_P4_H_
