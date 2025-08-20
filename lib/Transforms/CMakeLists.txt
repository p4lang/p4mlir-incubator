add_mlir_dialect_library(P4MLIRTransforms
  FlattenCFG.cpp
  PrintParsersGraph.cpp
  RemoveAliases.cpp
  SerEnumElimination.cpp
  SimplifyParsers.cpp
  SimplifySelect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Transforms

  DEPENDS
  P4MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
