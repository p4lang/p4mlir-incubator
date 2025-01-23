get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}

  P4MLIR_P4HIR

  MLIRFuncDialect
  MLIROptLib
)

set(P4MLIR_TRANSLATE_SRCS
  main.cpp
  options.cpp
  translate.cpp)

add_llvm_executable(p4mlir-translate ${P4MLIR_TRANSLATE_SRCS})
target_link_libraries(p4mlir-translate PRIVATE ${P4C_LIBRARIES} ${P4C_LIB_DEPS})

llvm_update_compile_flags(p4mlir-translate)
target_link_libraries(p4mlir-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(p4mlir-translate)

add_custom_target(linkp4mlir_translate_out
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${P4MLIR_BINARY_DIR}/bin/p4mlir-translate ${P4C_BINARY_DIR}/p4mlir-translate
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${P4C_BINARY_DIR}/p4include ${CMAKE_CURRENT_BINARY_DIR}/p4include
)

add_dependencies(p4c_driver linkp4mlir_translate_out)
