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
  UseControlPlaneNames.cpp
  FlattenStructs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Transforms

  DEPENDS
  P4MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
