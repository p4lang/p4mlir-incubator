#!/usr/bin/env bash
#
# Install common build tools

set -ex

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    ccache \
    clang \
    lld \
    ninja-build \
    pkg-config \
    cmake \
    python3 \
    python3-pip

# Install additional LLVM & MLIR dependencies
# https://llvm.org/docs/GettingStarted.html#requirements
# https://mlir.llvm.org/getting_started/
sudo apt-get install -y \
    zlib1g-dev

# Install additional P4C dependencies
# https://github.com/p4lang/p4c/blob/main/README.md#ubuntu-dependencies
sudo apt-get install -y \
    bison \
    flex \
    libboost-dev \
    libboost-iostreams-dev \
    libfl-dev
