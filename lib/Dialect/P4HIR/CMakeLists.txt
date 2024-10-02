add_mlir_dialect_library(P4MLIR_P4HIR
  P4HIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/P4HIR

  DEPENDS
  P4MLIR_P4HIR_IncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)
