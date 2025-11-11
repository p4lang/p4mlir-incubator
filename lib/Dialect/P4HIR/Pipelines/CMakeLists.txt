add_mlir_dialect_library(P4Pipelines
  CommonFrontEndPipeline.cpp

  LINK_LIBS PUBLIC
  MLIRPass
)
