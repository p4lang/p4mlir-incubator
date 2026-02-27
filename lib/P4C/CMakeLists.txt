set(LIBS
  P4MLIR_P4HIR
  
  MLIRLoopLikeInterface
)

set(P4MLIR_TRANSLATE_LIB_SRCS
  translate.cpp)

add_mlir_library(P4MLIR_P4C_TRANSLATE ${P4MLIR_TRANSLATE_LIB_SRCS} EXCLUDE_FROM_LIBMLIR DISABLE_INSTALL)

target_link_libraries(P4MLIR_P4C_TRANSLATE PRIVATE ${P4C_LIBRARIES} ${P4C_LIB_DEPS} p4mlir_utilities)
mlir_target_link_libraries(P4MLIR_P4C_TRANSLATE PRIVATE ${LIBS} p4mlir_utilities)
mlir_check_all_link_libraries(P4MLIR_P4C_TRANSLATE)
