set(LIBS
  MLIRTranslateLib
  P4MLIR_P4HIR
  P4MLIR_P4CoreLib
  P4MLIR_BMv2IR
  P4MLIRConversion
  P4MLIRTransforms
  P4MLIRConversion
  P4HIRToBMv2IR
  P4Pipelines
  BMv2Pipelines
  P4MLIR_TargetBMv2


  MLIRFuncDialect
  MLIROptLib
)

add_llvm_executable(p4mlir-to-json
    p4mlir-to-json.cpp
)

llvm_update_compile_flags(p4mlir-to-json)
mlir_target_link_libraries(p4mlir-to-json PRIVATE ${LIBS})

mlir_check_all_link_libraries(p4mlir-to-json)
