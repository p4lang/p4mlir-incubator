#!/usr/bin/env bash
#
# Reference:
# - https://mlir.llvm.org/getting_started/
# - https://llvm.org/docs/GettingStarted.html#requirements

set -ex

apt-get install -y \
    build-essential \
    ccache \
    clang \
    lld \
    ninja-build \
    python-is-python3 \
    python3 \
    python3-pip \
    zlib1g-dev

pip install --upgrade \
    cmake
