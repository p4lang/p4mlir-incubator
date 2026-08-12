# SPDX-FileCopyrightText: 2025 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name P4HIRToLLVM)
mlir_tablegen(Passes.cpp.inc -gen-pass-decls -name P4HIRToLLVM)
add_public_tablegen_target(P4HIRToLLVMPassIncGen)
