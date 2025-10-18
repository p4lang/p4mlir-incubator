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

/// @brief Exports a P4HIR MLIR module to a P4 source string.
/// @param module The MLIR module to export.
/// @param options Configuration options for the export process.
/// @return A string containing the P4 code on success, or failure.
llvm::FailureOr<std::string> exportP4HirToP4(mlir::ModuleOp module,
                                             P4HirToP4ExporterOptions options);

/// @brief Exports a P4HIR MLIR module to a P4 source file stream.
/// @param module The MLIR module to export.
/// @param os The output stream to write to.
/// @param options Configuration options for the export process.
/// @return Success or failure.
mlir::LogicalResult exportP4HirToP4(mlir::ModuleOp module, llvm::raw_ostream &os,
                                    P4HirToP4ExporterOptions options);

/// @brief Convenience function to export a P4HIR MLIR module directly to a file.
/// @param module The MLIR module to export.
/// @param p4OutputFile The path to the output P4 file.
/// @return Success or failure.
mlir::LogicalResult writeP4HirToP4File(mlir::ModuleOp module,
                                       const std::filesystem::path &p4OutputFile);

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_EXPORT_TO_P4_H_
