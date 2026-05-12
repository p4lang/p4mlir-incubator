/*
 * SPDX-FileCopyrightText: 2024 The P4 Language Consortium
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_DIALECT_H
