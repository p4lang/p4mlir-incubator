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

# Note that P4C uses both RTTI and C++ exceptions, so we need to build LLVM/MLIR having them enabled as well
cmake -G Ninja "$LLVM_REPO_DIR"/llvm \
   -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DLLVM_INCLUDE_BENCHMARKS=OFF \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_EH=ON

ninja
ninja check-mlir
ninja install
