// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDE_P4MLIR_P4C_DIAGNOSTICS_H_
#define INCLUDE_P4MLIR_P4C_DIAGNOSTICS_H_

#include <string>

#include "lib/source_file.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

namespace P4::P4MLIR {
class P4CDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
    struct DiagnosticAdaptor {
        mlir::Location loc;
        std::string msg;
        const P4::Util::InputSources *sources;

        DiagnosticAdaptor(mlir::Location loc, std::string msg,
                          const P4::Util::InputSources *sources)
            : loc(loc), msg(std::move(msg)), sources(sources) {}

        std::string toString() const { return msg; }

        P4::Util::SourceInfo getSourceInfo() const {
            if (auto fileLoc = loc->findInstanceOf<mlir::FileLineColLoc>()) {
                return {sources,
                        {fileLoc.getStartLine(), fileLoc.getStartColumn()},
                        {fileLoc.getEndLine(), fileLoc.getEndColumn()}};
            }

            return {};
        }
    };

    /// The maximum depth that a call stack will be printed.
    unsigned callStackLimit = 10;

 public:
    /// This type represents a functor used to filter out locations when printing
    /// a diagnostic. It should return true if the provided location is okay to
    /// display, false otherwise. If all locations in a diagnostic are filtered
    /// out, the first location is used as the sole location. When deciding
    /// whether or not to filter a location, this function should not recurse into
    /// any nested location. This recursion is handled automatically by the
    /// caller.
    using ShouldShowLocFn = llvm::unique_function<bool(mlir::Location, const mlir::Diagnostic &diag,
                                                       llvm::StringRef ctx)>;

    P4CDiagnosticHandler(mlir::MLIRContext *ctx, const P4::Util::InputSources *sources,
                         ShouldShowLocFn &&shouldShowLocFn = {})
        : mlir::ScopedDiagnosticHandler(ctx),
          ctx(ctx),
          sources(sources),
          shouldShowLocFn(std::move(shouldShowLocFn)) {
        setHandler([this](mlir::Diagnostic &diag) { emitDiagnostic(diag); });
    }

    ~P4CDiagnosticHandler() = default;

 protected:
    void emitDiagnostic(mlir::Location loc, llvm::Twine message, mlir::DiagnosticSeverity kind,
                        bool displaySourceLine = true);

    void emitDiagnostic(mlir::Diagnostic &diag);

 private:
    /// Given a location, returns the first nested location (including 'loc') that
    /// can be shown to the user.
    std::optional<mlir::Location> findLocToShow(mlir::Location loc, const mlir::Diagnostic &diag,
                                                llvm::StringRef ctx);

    mlir::MLIRContext *ctx;
    const P4::Util::InputSources *sources;
    ShouldShowLocFn shouldShowLocFn;
};

}  // namespace P4::P4MLIR

#endif  // INCLUDE_P4MLIR_P4C_DIAGNOSTICS_H_
