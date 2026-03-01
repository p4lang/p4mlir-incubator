add_mlir_dialect_library(P4HIRToCoreLib
  P4HIRToCoreLib.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Conversion/P4HIRToCoreLib

  DEPENDS
  P4HIRToCoreLibPassIncGen

  LINK_LIBS PUBLIC
  P4MLIRConversion
  P4MLIR_P4CoreLib
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRArithDialect
)