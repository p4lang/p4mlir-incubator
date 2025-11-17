add_mlir_dialect_library(P4Pipelines
  CommonFrontEndPipeline.cpp

  LINK_LIBS PUBLIC
  P4HIRToCoreLib
  P4MLIRTransforms
  MLIRPass
)
