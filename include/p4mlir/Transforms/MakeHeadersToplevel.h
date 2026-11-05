// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_TRANSFORMS_MAKE_HEADERS_TOPLEVEL_H
#define P4MLIR_TRANSFORMS_MAKE_HEADERS_TOPLEVEL_H

#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace P4::P4MLIR {

/// Apply the `make-headers-toplevel` pass transformation to the given object and arguments.
mlir::LogicalResult makeHeadersToplevel(mlir::FunctionOpInterface object,
                                        mlir::BlockArgument hdrsArg, mlir::BlockArgument metaArg);

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_MAKE_HEADERS_TOPLEVEL_H
