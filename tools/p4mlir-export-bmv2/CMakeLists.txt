get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  
  P4MLIR_P4HIR
  
  MLIRFuncDialect
  MLIROptLib
)

set(P4MLIR_EXPORT_SRCS
  p4mlir-export-bmv2.cpp
  options.cpp
  json_emitter.cpp
)

add_llvm_executable(p4mlir-export-bmv2 ${P4MLIR_EXPORT_SRCS})

target_link_libraries(p4mlir-export-bmv2 PRIVATE ${P4C_LIBRARIES} ${P4C_LIB_DEPS})

llvm_update_compile_flags(p4mlir-export-bmv2)

target_link_libraries(p4mlir-export-bmv2 PRIVATE ${LIBS})

mlir_check_all_link_libraries(p4mlir-export-bmv2)

add_custom_target(linkp4mlir_export_out
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
          ${P4MLIR_BINARY_DIR}/bin/p4mlir-export-bmv2
          ${P4C_BINARY_DIR}/p4mlir-export-bmv2
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
          ${P4C_BINARY_DIR}/p4include 
          ${CMAKE_CURRENT_BINARY_DIR}/p4include
)

add_dependencies(p4c_driver linkp4mlir_export_out)