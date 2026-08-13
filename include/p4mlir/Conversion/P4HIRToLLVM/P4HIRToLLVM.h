// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_CONVERSION_P4HIRTOLLVM_H
#define P4MLIR_CONVERSION_P4HIRTOLLVM_H

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace P4::P4MLIR {

#define GEN_PASS_DECL_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/P4HIRToLLVM/Passes.h.inc"

// Populates `converter` with P4HIR -> LLVM dialect type conversions. Exposed
// so that other conversion passes targeting the LLVM dialect can reuse the
// same type mapping instead of duplicating it.
void populateP4HIRToLLVMTypeConversion(mlir::LLVMTypeConverter &converter);

// Populates `patterns` with P4HIR -> LLVM dialect conversion patterns. Exposed
// so that other conversion passes targeting the LLVM dialect can reuse the
// same operation lowerings instead of duplicating them.
void populateP4HIRToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                           mlir::RewritePatternSet &patterns);

}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_P4HIRTOLLVM_H
