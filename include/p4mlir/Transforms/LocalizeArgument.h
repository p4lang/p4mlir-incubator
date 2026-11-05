// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_TRANSFORMS_LOCALIZE_ARGUMENT_H
#define P4MLIR_TRANSFORMS_LOCALIZE_ARGUMENT_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"

namespace P4::P4MLIR {

/// Assuming that `args` are arguments of top-level objects that all correspond to a single shared
/// resource, localize fields that are only used in one top-level object. Localized fields are moved
/// to variables within the corresponding top-level object. The original arguments are rewritten
/// to use a new type that only includes fields which may be shared between multiple objects. All of
/// `args` must be of the same reference type. Fields of the shared resource that may be accessed
/// outside of the given top-level objects need to be provided in `escapingFields` to prevent
/// incorrect localization. If `argNameTemplate` is non-empty it will be used to derive names for
/// the generated types and arguments.
mlir::LogicalResult localizeArg(llvm::ArrayRef<mlir::BlockArgument> args,
                                llvm::SmallVector<P4HIR::FieldPath> escapingFields = {},
                                llvm::StringRef argNameTemplate = "");

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_LOCALIZE_ARGUMENT_H
