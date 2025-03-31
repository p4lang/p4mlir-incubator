#ifndef LIB_UTILITIES_EXPORT_TO_P4_H_
#define LIB_UTILITIES_EXPORT_TO_P4_H_

#include <filesystem>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

#include "p4mlir/lib/Utilities/extended_formatted_stream.h"

namespace P4::P4MLIR::Utilities {

struct P4HirToP4ExporterOptions {
 public:
    /// The default indentation to use for the generated program (tabs vs spaces...).
    ExtendedFormattedOStream::IndentationStyle style;
    /// The default indentation level to use for the generated program.
    int indentLevel;
    /// If true, only the main package and associated programmable blocks will be exported.
    bool mainPackageOnly = false;

    P4HirToP4ExporterOptions()
        : style(ExtendedFormattedOStream::IndentationStyle::Spaces), indentLevel(4) {}
};

llvm::FailureOr<std::string> exportP4HirToP4(mlir::ModuleOp module,
                                             P4HirToP4ExporterOptions options);

mlir::LogicalResult exportP4HirToP4(mlir::ModuleOp module, llvm::raw_ostream &os,
                                    P4HirToP4ExporterOptions options);

mlir::LogicalResult writeP4HirToP4File(mlir::ModuleOp module,
                                       const std::filesystem::path &p4OutputFile);

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_EXPORT_TO_P4_H_
