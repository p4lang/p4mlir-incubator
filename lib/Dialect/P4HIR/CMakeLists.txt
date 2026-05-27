# SPDX-FileCopyrightText: 2024 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(P4MLIR_P4HIR
  P4HIR_Ops.cpp
  P4HIR_Types.cpp
  P4HIR_Attrs.cpp
  P4HIR_TypeInterfaces.cpp
  P4HIR_MemorySlot.cpp
  P4HIR_Symbols.cpp
  P4HIR_Mangle.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/P4HIR

  DEPENDS
  P4MLIR_P4HIR_IncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemorySlotInterfaces
  MLIRViewLikeInterface
  MLIRFuncDialect
)

add_subdirectory(Pipelines)
