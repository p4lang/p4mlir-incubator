# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(BMv2Pipelines
  BMv2Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRPass
)
