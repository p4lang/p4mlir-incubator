# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(P4Pipelines
  CommonFrontEndPipeline.cpp

  LINK_LIBS PUBLIC
  P4HIRToCoreLib
  P4MLIRTransforms
  MLIRPass
)
