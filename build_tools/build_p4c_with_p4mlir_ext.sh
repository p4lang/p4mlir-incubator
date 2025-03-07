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

P4C_REPO_DIR=$P4MLIR_REPO_DIR/third_party/p4c
P4C_BUILD_DIR=$P4C_REPO_DIR/build
P4C_EXT_DIR=$P4C_REPO_DIR/extensions

P4C_P4MLIR_EXT_DIR=$P4C_EXT_DIR/p4mlir

P4C_P4MLIR_EXT_SYMLINK=$(realpath --relative-to="$(dirname "$P4C_P4MLIR_EXT_DIR")" "$P4MLIR_REPO_DIR" 2>/dev/null || \
python3 -c 'import os, sys; print(os.path.relpath(sys.argv[2], start=sys.argv[1]))' "$(dirname "$P4C_P4MLIR_EXT_DIR")" "$P4MLIR_REPO_DIR")

# Link P4MLIR as P4C extension
mkdir -p "$P4C_EXT_DIR"
if [ ! -d "$P4C_P4MLIR_EXT_DIR" ]; then
    ln -s "$P4C_P4MLIR_EXT_SYMLINK" "$P4C_P4MLIR_EXT_DIR"
fi

# Configure CMake flags
CMAKE_FLAGS="-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER:-clang}"
CMAKE_FLAGS+=" -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER:-clang++}"

# Configure P4C CMake flags
# https://github.com/p4lang/p4c/blob/main/CMakeLists.txt
CMAKE_FLAGS+=" -DENABLE_DOCS=OFF"
CMAKE_FLAGS+=" -DENABLE_GTESTS=ON"
CMAKE_FLAGS+=" -DENABLE_BMV2=OFF"
CMAKE_FLAGS+=" -DENABLE_EBPF=OFF"
CMAKE_FLAGS+=" -DENABLE_UBPF=OFF"
CMAKE_FLAGS+=" -DENABLE_DPDK=OFF"
CMAKE_FLAGS+=" -DENABLE_TOFINO=OFF"
CMAKE_FLAGS+=" -DENABLE_P4TC=OFF"
CMAKE_FLAGS+=" -DENABLE_P4FMT=OFF"
CMAKE_FLAGS+=" -DENABLE_P4TEST=ON"
CMAKE_FLAGS+=" -DENABLE_TEST_TOOLS=OFF"
CMAKE_FLAGS+=" -DENABLE_P4C_GRAPHS=OFF"

# Configure P4MLIR CMake flags
CMAKE_FLAGS+=" -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR"
CMAKE_FLAGS+=" -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir"
CMAKE_FLAGS+=" -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit"

# Build P4C with P4MLIR extension
mkdir -p "$P4C_BUILD_DIR"
cd "$P4C_BUILD_DIR"
cmake -G Ninja "$P4C_REPO_DIR" $CMAKE_FLAGS
ninja

# Run some tests
ninja check-p4mlir

# Install
ninja install
