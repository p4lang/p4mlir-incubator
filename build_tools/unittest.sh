#!/usr/bin/env bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "SCRIPT_DIR: $SCRIPT_DIR"

SCRIPT_DIR="${GITHUB_WORKSPACE}/build_tools"

echo "SCRIPT_DIR: $SCRIPT_DIR"

P4MLIR_REPO_DIR=$( cd "$SCRIPT_DIR"/.. &> /dev/null && pwd )

P4C_REPO_DIR=$P4MLIR_REPO_DIR/third_party/p4c
P4C_BUILD_DIR=$P4C_REPO_DIR/build
P4C_EXT_DIR=$P4C_REPO_DIR/extensions
P4C_P4MLIR_EXT_DIR=$P4C_EXT_DIR/p4mlir
LLVM_LIT="$P4MLIR_REPO_DIR/third_party/llvm-project/build/bin/llvm-lit"
TEST_DIR="$P4MLIR_REPO_DIR/third_party/p4c/build/extensions/p4mlir/test"

cd "$P4C_BUILD_DIR"
echo "*********** Listing all available tests ***************"
TEST_LIST=$($LLVM_LIT --show-tests "$TEST_DIR")

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

TESTS=()
while IFS= read -r line; do
    TESTS+=("$line")
done < <($LLVM_LIT --show-tests "$TEST_DIR" | grep "P4MLIR ::" | sed 's/^  P4MLIR :: //')

for TEST_ITEM in "${TESTS[@]}"; do

    FULL_TEST_PATH="$TEST_DIR/$TEST_ITEM"
    
    echo "Test Item: $TEST_ITEM"
    echo "Full Test Path: $FULL_TEST_PATH"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if $LLVM_LIT -v "$FULL_TEST_PATH"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "PASSED"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "FAILED"
    fi
done


echo "==== Test Summary ===="
echo "Total tests: $TOTAL_TESTS"
echo "Passed tests: $PASSED_TESTS"
echo "Failed tests: $FAILED_TESTS"

echo "==== Installing ===="
ninja install
