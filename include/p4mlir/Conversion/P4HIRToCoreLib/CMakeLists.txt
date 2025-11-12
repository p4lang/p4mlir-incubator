set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name P4MLIRConversion)
mlir_tablegen(Passes.cpp.inc -gen-pass-decls -name P4MLIRConversion)
add_public_tablegen_target(P4MLIRConversionPassIncGen)
