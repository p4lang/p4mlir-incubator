# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(P4MLIR_TargetBMv2
  Target.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Targets/BMv2

  LINK_LIBS PUBLIC
  P4MLIR_BMv2IR
)
