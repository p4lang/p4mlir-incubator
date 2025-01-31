set(P4MLIR_TRANSLATE_SRCS
    main.cpp options.cpp)

add_executable(p4mlir-translate ${P4MLIR_TRANSLATE_SRCS})
target_link_libraries(p4mlir-translate ${P4C_LIBRARIES} ${P4C_LIB_DEPS})
llvm_update_compile_flags(p4mlir-translate)

add_custom_target(linkp4mlir_translate_out
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_BINARY_DIR}/p4mlir-translate ${P4C_BINARY_DIR}/p4mlir-translate
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${P4C_BINARY_DIR}/p4include ${CMAKE_CURRENT_BINARY_DIR}/p4include
)

add_dependencies(p4c_driver linkp4mlir_translate_out)
