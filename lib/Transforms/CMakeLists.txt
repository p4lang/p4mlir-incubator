# Compile PDLL patterns using mlir-pdll
set(PDLL_SOURCE ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/P4HIR/StrengthReduction.pdll)
set(PDLL_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/StrengthReductionPatterns.h.inc)

add_custom_command(
  OUTPUT ${PDLL_OUTPUT}
  COMMAND mlir-pdll ${PDLL_SOURCE} -x=cpp -o ${PDLL_OUTPUT}
          -I ${PROJECT_SOURCE_DIR}/include
          -I ${PROJECT_SOURCE_DIR}/include/p4mlir/Dialect/P4HIR
          -I ${MLIR_INCLUDE_DIRS}
  DEPENDS ${PDLL_SOURCE}
  COMMENT "Building StrengthReductionPatterns.h.inc from PDLL"
  VERBATIM
)

add_custom_target(P4MLIR_StrengthReductionPatternsIncGen DEPENDS ${PDLL_OUTPUT})

# Add build directory to include paths for generated PDLL headers
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_dialect_library(P4MLIRTransforms
  EnumElimination.cpp
  FlattenCFG.cpp
  IRUtils.cpp
  InlineParserControl.cpp
  PrintParsersGraph.cpp
  RemoveAliases.cpp
  RemoveSoftCF.cpp
  SerEnumElimination.cpp
  SimplifyParsers.cpp
  CopyInCopyOutElimination.cpp
  TupleToStruct.cpp
  ExpandEmit.cpp
  SymbolDCE.cpp
  StrengthReduction.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Transforms
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  P4MLIRTransformsPassIncGen
  P4MLIR_StrengthReductionPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRPDLDialect
  MLIRPDLInterpDialect
  P4MLIR_P4HIR
)
