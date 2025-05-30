add_mlir_dialect_library(P4MLIRConversion
  P4HIRToCoreLib.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Conversion

  DEPENDS
  P4MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
