// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Location.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/TypeSwitch.h"
#include "p4mlir/P4C/diagnostics.h"
#pragma GCC diagnostic pop

#include "lib/compile_context.h"
#include "lib/error_catalog.h"
#include "lib/error_reporter.h"

using namespace P4::P4MLIR;

static P4::DiagnosticAction getDefaultAction(mlir::DiagnosticSeverity kind) {
    auto &context = P4::BaseCompileContext::get();

    switch (kind) {
        case mlir::DiagnosticSeverity::Note:
        case mlir::DiagnosticSeverity::Remark:
            return context.getDefaultInfoDiagnosticAction();
        case mlir::DiagnosticSeverity::Warning:
            return context.getDefaultWarningDiagnosticAction();
        case mlir::DiagnosticSeverity::Error:
            return context.getDefaultErrorDiagnosticAction();
    }
    llvm_unreachable("Unknown DiagnosticSeverity");
}

static int getErrorCode(mlir::DiagnosticSeverity kind) {
    switch (kind) {
        case mlir::DiagnosticSeverity::Note:
        case mlir::DiagnosticSeverity::Remark:
            return P4::ErrorType::INFO_INFERRED;
        case mlir::DiagnosticSeverity::Warning:
            return P4::ErrorType::WARN_UNSUPPORTED;
        case mlir::DiagnosticSeverity::Error:
            return P4::ErrorType::ERR_UNSUPPORTED_ON_TARGET;
    }
    llvm_unreachable("Unknown DiagnosticSeverity");
}

/// Return a processable CallSiteLoc from the given location.
static std::optional<mlir::CallSiteLoc> getCallSiteLoc(mlir::Location loc) {
    return llvm::TypeSwitch<mlir::LocationAttr, std::optional<mlir::CallSiteLoc>>(loc)
        .Case([](mlir::NameLoc nameLoc) { return getCallSiteLoc(nameLoc.getChildLoc()); })
        .Case([](mlir::CallSiteLoc callLoc) { return callLoc; })
        .Case([](mlir::FusedLoc fusedLoc) -> std::optional<mlir::CallSiteLoc> {
            for (auto subLoc : fusedLoc.getLocations()) {
                if (auto callLoc = getCallSiteLoc(subLoc)) {
                    return callLoc;
                }
            }
            return std::nullopt;
        })
        .Default([](auto) -> std::optional<mlir::CallSiteLoc> { return std::nullopt; });
}

void P4CDiagnosticHandler::emitDiagnostic(mlir::Location loc, llvm::Twine message,
                                          mlir::DiagnosticSeverity kind, bool displaySourceLine) {
    auto &context = P4::BaseCompileContext::get();
    // Grab default action for this severity kind
    auto action = getDefaultAction(kind);

    DiagnosticAdaptor p4cDiag(displaySourceLine ? loc : mlir::UnknownLoc::get(ctx), message.str(),
                              sources);
    // Note that error reporter does duplicate diagnostic filtering only for
    // objects passed by address, not by reference (as it expects `T->getSourceInfo()` to be
    // valid).
    context.errorReporter().diagnose(action, getErrorCode(kind), "%1%", "", &p4cDiag);
}

void P4CDiagnosticHandler::emitDiagnostic(mlir::Diagnostic &diag) {
    auto loc = diag.getLocation();
    llvm::SmallVector<std::pair<mlir::Location, llvm::StringRef>> locationStack;

    auto maybeAddLocToStack = [&](mlir::Location loc, llvm::StringRef locContext) {
        if (auto showableLoc = findLocToShow(loc, diag, locContext))
            locationStack.emplace_back(*showableLoc, locContext);
    };

    // Add locations to display for this diagnostic.
    maybeAddLocToStack(loc, "");

    // If the diagnostic location was a call site location, add the call stack as
    // well.
    if (auto callLoc = getCallSiteLoc(loc)) {
        // Print the call stack while valid, or until the limit is reached.
        loc = callLoc->getCaller();
        for (unsigned curDepth = 0; curDepth < callStackLimit; ++curDepth) {
            maybeAddLocToStack(loc, "called from");
            if ((callLoc = getCallSiteLoc(loc)))
                loc = callLoc->getCaller();
            else
                break;
        }
    }

    // If the location stack is empty, use the initial location (unless the entire diagnostics is
    // filtered out)
    if (locationStack.empty()) {
        if (!shouldShowLocFn || shouldShowLocFn(diag.getLocation(), diag, ""))
            emitDiagnostic(diag.getLocation(), diag.str(), diag.getSeverity());

        // Otherwise, use the location stack.
    } else {
        emitDiagnostic(locationStack.front().first, diag.str(), diag.getSeverity());
        for (auto &it : llvm::drop_begin(locationStack))
            emitDiagnostic(it.first, it.second, mlir::DiagnosticSeverity::Note);
    }

    // Emit each of the notes. Only display the source code if the location is
    // different from the previous location.
    for (auto &note : diag.getNotes()) {
        emitDiagnostic(note.getLocation(), note.str(), note.getSeverity(),
                       /*displaySourceLine=*/loc != note.getLocation());
        loc = note.getLocation();
    }
}

std::optional<mlir::Location> P4CDiagnosticHandler::findLocToShow(mlir::Location loc,
                                                                  const mlir::Diagnostic &diag,
                                                                  llvm::StringRef ctx) {
    if (!shouldShowLocFn) return loc;
    if (!shouldShowLocFn(loc, diag, ctx)) return std::nullopt;

    // Recurse into the child locations of some of location types.
    return llvm::TypeSwitch<mlir::LocationAttr, std::optional<mlir::Location>>(loc)
        .Case([&](mlir::CallSiteLoc callLoc) -> std::optional<mlir::Location> {
            // We recurse into the callee of a call site, as the caller will be
            // emitted in a different note on the main diagnostic.
            return findLocToShow(callLoc.getCallee(), diag, ctx);
        })
        .Case([&](mlir::FileLineColLoc) -> std::optional<mlir::Location> { return loc; })
        .Case([&](mlir::FusedLoc fusedLoc) -> std::optional<mlir::Location> {
            // Fused location is unique in that we try to find a sub-location to
            // show, rather than the top-level location itself.
            for (auto childLoc : fusedLoc.getLocations())
                if (auto showableLoc = findLocToShow(childLoc, diag, ctx)) return showableLoc;
            return std::nullopt;
        })
        .Case([&](mlir::NameLoc nameLoc) -> std::optional<mlir::Location> {
            return findLocToShow(nameLoc.getChildLoc(), diag, ctx);
        })
        .Case([&](mlir::OpaqueLoc opaqueLoc) -> std::optional<mlir::Location> {
            // OpaqueLoc always falls back to a different source location.
            return findLocToShow(opaqueLoc.getFallbackLocation(), diag, ctx);
        })
        .Case([](mlir::UnknownLoc) -> std::optional<mlir::Location> {
            // Prefer not to show unknown locations.
            return std::nullopt;
        });
}
