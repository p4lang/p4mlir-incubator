configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(P4MLIR_TEST_DEPENDS
  FileCheck count not
  p4mlir-opt
  p4mlir-translate
)

add_lit_testsuite(check-p4mlir "Running the P4MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${P4MLIR_TEST_DEPENDS}
)
set_target_properties(check-p4mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(P4MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${P4MLIR_TEST_DEPENDS})
