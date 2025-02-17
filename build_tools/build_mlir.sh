#!/usr/bin/env bash
#
# Reference:
# - https://mlir.llvm.org/getting_started/
# - https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone

set -ex

# https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )  # p4mlir/build_tools
P4MLIR_REPO_DIR=$( cd "$SCRIPT_DIR"/.. &> /dev/null && pwd )

LLVM_REPO_DIR=$P4MLIR_REPO_DIR/third_party/llvm-project
LLVM_BUILD_DIR=$LLVM_REPO_DIR/build
LLVM_INSTALL_DIR=$P4MLIR_REPO_DIR/install

mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

# Configure CMake flags
# Note that P4C uses both RTTI and C++ exceptions, so we need to build LLVM/MLIR having them enabled as well
CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR"
CMAKE_FLAGS+=" -DLLVM_ENABLE_PROJECTS=mlir"
CMAKE_FLAGS+=" -DLLVM_BUILD_EXAMPLES=OFF"
CMAKE_FLAGS+=" -DLLVM_TARGETS_TO_BUILD="Native""
CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"
CMAKE_FLAGS+=" -DLLVM_ENABLE_ASSERTIONS=ON"
CMAKE_FLAGS+=" -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER:-clang}"
CMAKE_FLAGS+=" -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER:-clang++}"
if [[ -n "${LLVM_USE_LINKER}" ]]; then
    CMAKE_FLAGS+=" -DLLVM_USE_LINKER=${LLVM_USE_LINKER}"
else
    CMAKE_FLAGS+=" -DLLVM_ENABLE_LLD=ON"
fi
CMAKE_FLAGS+=" -DLLVM_CCACHE_BUILD=ON"
CMAKE_FLAGS+=" -DLLVM_INSTALL_UTILS=ON"
CMAKE_FLAGS+=" -DLLVM_INCLUDE_BENCHMARKS=OFF"
CMAKE_FLAGS+=" -DLLVM_ENABLE_RTTI=ON"
CMAKE_FLAGS+=" -DLLVM_ENABLE_EH=ON"

cmake -G Ninja "$LLVM_REPO_DIR"/llvm $CMAKE_FLAGS

ninja
ninja check-mlir
ninja install
