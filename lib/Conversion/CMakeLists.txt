add_mlir_dialect_library(P4MLIRConversion
  ConversionPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Conversion

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
)

add_subdirectory(P4HIRToCoreLib)
add_subdirectory(P4HIRToBMv2IR)
