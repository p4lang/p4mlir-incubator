add_mlir_dialect_library(P4MLIR_BMv2IR
  BMv2IR_Ops.cpp
  BMv2IR_Types.cpp
  BMv2IR_Attrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/BMv2IR

  DEPENDS
  P4MLIR_BMv2IR_IncGen
  P4MLIR_BMv2IR_AttrIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)
