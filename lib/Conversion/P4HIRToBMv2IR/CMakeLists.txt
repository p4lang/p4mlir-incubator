# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(P4HIRToBMv2IR
  P4HIRToBMv2IR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Conversion/P4HIRToBMv2IR

  DEPENDS
  P4HIRToBMv2IRPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
