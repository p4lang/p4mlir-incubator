# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(P4MLIR_P4CoreLib
  P4CoreLib_Ops.cpp
  P4CoreLib_Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/P4CoreLib

  DEPENDS
  P4MLIR_P4CoreLib_IncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)
