set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name P4HIRToBMv2IR)
mlir_tablegen(Passes.cpp.inc -gen-pass-decls -name P4HIRToBMv2IR)
add_public_tablegen_target(P4HIRToBMv2IRPassIncGen)
